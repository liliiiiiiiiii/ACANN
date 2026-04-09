"""
ACANN_v3: Gaussian Mixture Parameter Prediction for 2D Spectral Function.

核心思想
--------
谱函数 A(ω₁,ω₂) 是 K 个二维高斯的归一化混合。与其用 CNN 逐像素重建
1024×1024 网格，不如直接预测 K 组高斯参数 (μ₁, μ₂, σ₁, σ₂, amp)，
再解析渲染。预测 ~5K 个参数远比解码 100 万像素更高效、更精确。

架构
----
Input  : (B, nb_gc)  切比雪夫系数
Output : (B, w1, w2) 归一化谱函数

1. InputNorm → Linear projection → d_hidden
2. N × ResMLPBlock（Pre-LN + GELU + Dropout + Skip）
3. FinalNorm → ParameterHead → (μ₁, μ₂, σ₁, σ₂, amp, gate) × K
4. 可分离高斯渲染（memory-efficient einsum）
5. 物理归一化: ∫∫A dω₁dω₂ = 1

优势
----
- 输出受限为高斯混合 → 不可能产生模糊斑块、棋盘伪影
- 分辨率无关：同一组参数可渲染到任意分辨率
- 参数量 ~6M（远小于 v2 的 Transformer+CNN）
- 可分离渲染：O(K·(W₁+W₂)) 中间张量 → 显存极低

参考
----
- Physics-informed neural networks: Raissi et al., J. Comput. Phys. 2019
- Gaussian mixture regression: Bishop, "Pattern Recognition and ML" 2006
- Pre-LN Transformer: Xiong et al., ICML 2020
- GELU activation: Hendrycks & Gimpel, 2016
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResMLPBlock(nn.Module):
    """Pre-LN 残差 MLP 块: LN → Linear → GELU → Drop → Linear → Drop + skip。"""

    def __init__(self, dim: int, expansion: int = 2, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ACANN_v3(nn.Module):
    """
    Parameters
    ----------
    nb_gc        : 切比雪夫系数个数（输入维度）
    n_peaks      : 高斯峰数量
    w1, w2       : 输出谱函数空间尺寸
    omega1_range : ω₁ 网格 (start, end)
    omega2_range : ω₂ 网格 (start, end)
    d_hidden     : MLP 隐藏层维度
    n_layers     : 残差 MLP 层数
    dropout      : Dropout 概率
    dw1, dw2     : 积分归一化网格步长
    """

    def __init__(
        self,
        nb_gc: int = 128,
        n_peaks: int = 3,
        w1: int = 500,
        w2: int = 500,
        omega1_range: tuple = (0.005, 4.995),
        omega2_range: tuple = (0.005, 4.995),
        d_hidden: int = 512,
        n_layers: int = 6,
        dropout: float = 0.05,
        dw1: float = 1.0,
        dw2: float = 1.0,
    ):
        super().__init__()
        self.n_peaks = n_peaks
        self.w1, self.w2 = w1, w2
        self.dw1, self.dw2 = dw1, dw2
        self.omega1_max = omega1_range[1]

        self.register_buffer(
            "omega1", torch.linspace(omega1_range[0], omega1_range[1], w1)
        )
        self.register_buffer(
            "omega2", torch.linspace(omega2_range[0], omega2_range[1], w2)
        )

        # ── 编码器 ─────────────────────────────────────────────────────────
        self.input_norm = nn.LayerNorm(nb_gc)
        self.input_proj = nn.Linear(nb_gc, d_hidden)
        self.blocks = nn.ModuleList(
            [ResMLPBlock(d_hidden, expansion=2, dropout=dropout) for _ in range(n_layers)]
        )
        self.final_norm = nn.LayerNorm(d_hidden)

        # ── 参数预测头 ─────────────────────────────────────────────────────
        # 每个峰 6 个参数: mu1, mu2, sigma1, sigma2, amp, gate
        self.param_head = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, n_peaks * 6),
        )

        self._init_weights()

    # ───────────────────────────────────────────────────────────────────────
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # 残差块输出层极小初始化 → 初始时每块近似恒等映射，但梯度仍可流通
        for blk in self.blocks:
            nn.init.normal_(blk.net[-2].weight, std=1e-4)
            nn.init.zeros_(blk.net[-2].bias)

        last = self.param_head[-1]
        nn.init.normal_(last.weight, std=1e-4)
        with torch.no_grad():
            bias = last.bias
            default_mus = [0.8, 1.1, 1.4]# 初始化[0.8, 1.1, 1.4]
            for k in range(self.n_peaks):
                i = k * 6
                tgt = default_mus[min(k, len(default_mus) - 1)]
                bias[i + 0] = math.log(tgt / max(2.2 - tgt, 1e-3))
                bias[i + 1] = 0.0
                p = (0.15 - 0.02) / (0.3 - 0.02)
                bias[i + 2] = math.log(p / (1 - p))
                bias[i + 3] = math.log(0.015)
                bias[i + 4] = 0.5               # amp ≈ softplus(0.5)+0.01 ≈ 1.0
                bias[i + 5] = 2.0               # gate ≈ sigmoid(2) ≈ 0.88
                # default_mus = [0.8, 1.1, 1.4]  # 初始化[0.8, 1.1, 1.4]
                # tgt_mu1 = default_mus[k]
                # p_mu = tgt_mu1 / 3
                # bias[i + 0] = math.log(tgt_mu1 / max(3.0 - tgt_mu1, 1e-3))
                # tgt_mu2 = default_mus[k]
                # bias[i + 1] = math.log(tgt_mu2 / max(3.0 - tgt_mu2, 1e-3))
                # p = (0.15 - 0.02) / (0.3 - 0.02)
                # bias[i + 2] = math.log(p / (1 - p))  # sigma1_init ≈ 0.15
                # bias[i + 3] = math.log(p / (1 - p))  # sigma2_init ≈ 0.015
                # bias[i + 4] = 0.5  # amp_init ≈ softplus(0.5)+0.01 ≈ 1.0
                # bias[i + 5] = 2.0  # gate ≈ sigmoid(2) ≈ 0.88

    # ───────────────────────────────────────────────────────────────────────
    def decode_params(self, raw: torch.Tensor):
        """raw (B, K, 6) → 物理约束参数 + 峰存在门控。"""
        mu1    = torch.sigmoid(raw[..., 0]) * 2.2 # [0.0-2.2]
        mu2    = torch.tanh(raw[..., 1]) * 0.1                # [-0.1, 0.1]
        sigma1 = 0.02 + (0.3 - 0.02) * torch.sigmoid(raw[..., 2]) # [0.02-0.3]
        sigma2 = torch.exp(raw[..., 3].clamp(-8, -2))         # [0.00034, 0.135]
        amp    = F.softplus(raw[..., 4]) + 0.01               # > 0
        gate   = torch.sigmoid(raw[..., 5])       # 峰存在概率 [0, 1]
        amp    = amp * gate                        # 非活跃峰可衰减至 ~0
        return mu1, mu2, sigma1, sigma2, amp, gate

    # ───────────────────────────────────────────────────────────────────────
    def render(self, mu1, mu2, sigma1, sigma2, amp):
        """
        可分离高斯渲染: A(ω₁,ω₂) = Σ_k c_k · exp(-½((ω₁-μ₁)/σ₁)²) · exp(-½((ω₂-μ₂)/σ₂)²)

        利用可分离结构，避免实例化 (B, K, W1, W2) 张量。
        """
        d1 = (self.omega1[None, None, :] - mu1[..., None]) / sigma1[..., None]#(B, K, w1)
        g1 = torch.exp(-0.5 * d1 * d1)        # (B, K, W1)

        d2 = (self.omega2[None, None, :] - mu2[..., None]) / sigma2[..., None]
        g2 = torch.exp(-0.5 * d2 * d2)        # (B, K, W2)

        coeff = amp / (2.0 * math.pi * sigma1 * sigma2)    # (B, K)

        # einsum: Σ_k coeff_k · g1_k ⊗ g2_k → (B, W1, W2)
        A = torch.einsum("bkw,bkv->bwv", coeff[..., None] * g1, g2)
        return A

    # ───────────────────────────────────────────────────────────────────────
    def _forward_impl(self, x: torch.Tensor):
        """编码 → 解码 → 排序 → 渲染 → 归一化，返回 (A, params)。"""
        B = x.size(0)
        h = self.input_proj(self.input_norm(x))
        for blk in self.blocks:
            h = blk(h)
        raw = self.param_head(self.final_norm(h)).view(B, self.n_peaks, 6)
        mu1, mu2, s1, s2, amp, gate = self.decode_params(raw)

        idx = mu1.argsort(dim=1)
        mu1, mu2, s1, s2, amp, gate = (
            t.gather(1, idx) for t in (mu1, mu2, s1, s2, amp, gate)
        )

        A = self.render(mu1, mu2, s1, s2, amp)
        A = A / (A.sum(dim=(1, 2), keepdim=True) * self.dw1 * self.dw2 + 1e-12)
        return A, dict(mu1=mu1, mu2=mu2, sigma1=s1, sigma2=s2, amp=amp, gate=gate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)[0]

    def forward_with_params(self, x: torch.Tensor):
        """返回 (A, params_dict)，用于训练和评估。"""
        return self._forward_impl(x)

    # ───────────────────────────────────────────────────────────────────────
    def sigma_regularization(
        self, sigma1, sigma2,
        s1_range=(0.02, 0.35), s2_range=(0.002, 0.015),
    ):
        """将 σ 拉入数据范围的正则项（范围外二次惩罚，范围内零惩罚）。"""
        log_s1 = torch.log(sigma1)
        lo1, hi1 = math.log(s1_range[0]), math.log(s1_range[1])
        r1 = F.relu(lo1 - log_s1).pow(2) + F.relu(log_s1 - hi1).pow(2)

        log_s2 = torch.log(sigma2)
        lo2, hi2 = math.log(s2_range[0]), math.log(s2_range[1])
        r2 = F.relu(lo2 - log_s2).pow(2) + F.relu(log_s2 - hi2).pow(2)

        return r1.mean() + r2.mean()

    # ───────────────────────────────────────────────────────────────────────
    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
