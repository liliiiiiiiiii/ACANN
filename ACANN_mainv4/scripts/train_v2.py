"""
ACANN_v3 训练脚本

改进一览（相比旧版）
──────────────────────────────────────────────────────
1. 配合 ACANN_v3 高斯参数预测模型
2. 多尺度损失        MSE + L₁（可选多尺度平均池化，保证远距离梯度信号）
3. EMA 权重          验证/推理使用 EMA 模型
4. 混合精度          torch.amp.autocast + GradScaler（损失在 float32 下计算）
5. 梯度裁剪          防止梯度爆炸
6. 余弦退火 + warmup OneCycleLR
7. 最优 checkpoint   保存 val_loss 最低的模型
8. 断点续训          --resume

运行示例
──────────────────────────────────────────────────────
  python scripts/train_v2.py
  python scripts/train_v2.py --config config/train_config.ini
  python scripts/train_v2.py --resume checkpoints/best.pth
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import argparse
import configparser
import logging
from typing import Optional
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split

from ACANN_mainv4.models.acann_v2 import ACANN_v3
from ACANN_mainv4.data.dataset import Database
from ACANN_mainv4.utils.rolling_mean import rolling_mean


# ─────────────────────────────────────────────────────────────────────────────
# EMA
# ─────────────────────────────────────────────────────────────────────────────

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9995):
        self.model  = model
        self.decay  = decay
        self.shadow = {
            k: v.clone().detach()
            for k, v in model.named_parameters() if v.requires_grad
        }

    @torch.no_grad()
    def update(self):
        d = self.decay
        for k, v in self.model.named_parameters():
            if v.requires_grad:
                self.shadow[k].mul_(d).add_(v.detach(), alpha=1.0 - d)

    class _ApplyContext:
        def __init__(self, ema):
            self._ema    = ema
            self._backup = {}

        def __enter__(self):
            for k, v in self._ema.model.named_parameters():
                if v.requires_grad:
                    self._backup[k] = v.data.clone()
                    v.data.copy_(self._ema.shadow[k])

        def __exit__(self, *_):
            for k, v in self._ema.model.named_parameters():
                if v.requires_grad:
                    v.data.copy_(self._backup[k])

    def apply(self):
        return self._ApplyContext(self)


# ─────────────────────────────────────────────────────────────────────────────
# 损失函数
# ─────────────────────────────────────────────────────────────────────────────

def combined_loss(
    A_pred: torch.Tensor,
    A_true: torch.Tensor,
    mse_w: float = 1.0,
    l1_w: float  = 0.5,
):
    """
    多尺度 MSE + L₁。

    4 个尺度 (1, 4, 16, 64)：
    - scale=1: 像素精度
    - scale=4/16: 中等尺度结构
    - scale=64: 16×16 分辨率，提供强长程位置梯度
    """
    total = torch.tensor(0.0, device=A_pred.device)
    sub = {}

    for s in [1, 4, 16, 64]:
        if s > 1:
            Ap = F.avg_pool2d(A_pred.unsqueeze(1), s).squeeze(1)
            At = F.avg_pool2d(A_true.unsqueeze(1), s).squeeze(1)
        else:
            Ap, At = A_pred, A_true
        mse = F.mse_loss(Ap, At)
        l1  = F.l1_loss(Ap, At)
        total = total + mse_w * mse + l1_w * l1
        sub[f"mse_s{s}"] = mse.item()
        sub[f"l1_s{s}"]  = l1.item()

    total = total / 4.0
    return total, sub


# ─────────────────────────────────────────────────────────────────────────────
# 配置读取
# ─────────────────────────────────────────────────────────────────────────────

def read_config(path: str) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    cfg.read(path, encoding="utf-8")
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# 主训练函数
# ─────────────────────────────────────────────────────────────────────────────

def train(config_path: str, resume_path: Optional[str] = None):
    cfg = read_config(config_path)

    # ── [data] ───────────────────────────────────────────────────────────────
    train_meta  = str(ROOT / cfg.get("data", "train_meta"))
    train_nc    = str(ROOT / cfg.get("data", "train_nc"))
    val_meta    = str(ROOT / cfg.get("data", "val_meta"))
    val_nc      = str(ROOT / cfg.get("data", "val_nc"))
    nb_train    = cfg.getint("data", "nb_train",  fallback=5000)
    nb_val      = cfg.getint("data", "nb_val",    fallback=500)
    train_w1    = cfg.getint("data", "train_w1",  fallback=1024)
    train_w2    = cfg.getint("data", "train_w2",  fallback=1024)

    same_dataset = (train_meta == val_meta and train_nc == val_nc)

    # ── [model] ──────────────────────────────────────────────────────────────
    nb_gc    = cfg.getint  ("model", "nb_gc",    fallback=64)
    n_peaks  = cfg.getint  ("model", "n_peaks",  fallback=3)
    d_hidden = cfg.getint  ("model", "d_hidden", fallback=512)
    n_layers = cfg.getint  ("model", "n_layers", fallback=6)
    dropout  = cfg.getfloat("model", "dropout",  fallback=0.05)

    # ── [optim] ──────────────────────────────────────────────────────────────
    lr             = cfg.getfloat("optim", "lr",             fallback=1e-3)
    lr_min         = cfg.getfloat("optim", "lr_min",         fallback=1e-6)
    weight_decay   = cfg.getfloat("optim", "weight_decay",   fallback=0.01)
    grad_clip      = cfg.getfloat("optim", "grad_clip",      fallback=1.0)
    warmup_epochs  = cfg.getint  ("optim", "warmup_epochs",  fallback=5)
    epochs         = cfg.getint  ("optim", "epochs",         fallback=200)
    batch_size     = cfg.getint  ("optim", "batch_size",     fallback=128)
    num_workers    = cfg.getint  ("optim", "num_workers",    fallback=4)
    use_amp        = cfg.getboolean("optim", "amp",          fallback=True)
    use_ema        = cfg.getboolean("optim", "ema",          fallback=True)
    ema_decay      = cfg.getfloat("optim", "ema_decay",      fallback=0.9995)

    # ── [loss] ───────────────────────────────────────────────────────────────
    mse_w  = cfg.getfloat("loss", "mse_w",  fallback=1.0)
    l1_w   = cfg.getfloat("loss", "l1_w",   fallback=0.5)
    gate_w = cfg.getfloat("loss", "gate_w", fallback=0.02)

    # ── [output] ─────────────────────────────────────────────────────────────
    ckpt_dir    = str(ROOT / cfg.get("output", "ckpt_dir",  fallback="checkpoints"))
    plot_dir    = str(ROOT / cfg.get("output", "plot_dir",  fallback="plots"))
    print_every = cfg.getint("output", "print_every", fallback=50)
    val_every   = cfg.getint("output", "val_every",   fallback=200)

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # ── 日志 ──────────────────────────────────────────────────────────────────
    log_path = os.path.join(ckpt_dir, "train.log")
    log = logging.getLogger("acann_v3")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    log.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    log.addHandler(sh)

    # ── 设备 & 精度 ──────────────────────────────────────────────────────────
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp   = use_amp and device.type == "cuda"
    amp_dtype = torch.float16 if use_amp else torch.float32
    scaler    = torch.cuda.amp.GradScaler(enabled=use_amp)
    log.info(f"Device: {device}  AMP: {use_amp}  EMA: {use_ema}")

    # ── 从 meta 读取 omega 网格和积分步长 ────────────────────────────────────
    _meta_path = Path(train_meta)
    o1_start, o1_end = 0.005, 4.995
    o2_start, o2_end = 0.005, 4.995
    dw1, dw2 = (o1_end-o1_start) / train_w1,(o2_end-o2_start) / train_w2

    if _meta_path.exists():
        _meta = np.load(str(_meta_path), allow_pickle=True)
        if "omega1" in _meta.files and _meta["omega1"] is not None:
            _o1 = _meta["omega1"].astype(np.float64)
            if len(_o1) > 1:
                o1_start, o1_end = float(_o1[0]), float(_o1[-1])
                dw1 = float(_o1[-1] - _o1[0]) / max(train_w1 - 1, 1)
        if "omega2" in _meta.files and _meta["omega2"] is not None:
            _o2 = _meta["omega2"].astype(np.float64)
            if len(_o2) > 1:
                o2_start, o2_end = float(_o2[0]), float(_o2[-1])
                dw2 = float(_o2[-1] - _o2[0]) / max(train_w2 - 1, 1)

    log.info(f"Omega1: [{o1_start:.4f}, {o1_end:.4f}],  Omega2: [{o2_start:.4f}, {o2_end:.4f}]")
    log.info(f"Grid spacing (for {train_w1}x{train_w2}): dw1={dw1:.6f}  dw2={dw2:.6f}")

    # ── 数据集 ───────────────────────────────────────────────────────────────
    target = (train_w1, train_w2)
    log.info(f"加载数据集，训练分辨率 {target} ...")

    if same_dataset:
        full_n   = nb_train + nb_val
        full_ds  = Database(train_meta, train_nc, nb_data=full_n, target_size=target)
        train_ds, val_ds = random_split(
            full_ds, [nb_train, min(nb_val, len(full_ds) - nb_train)],
            generator=torch.Generator().manual_seed(42),
        )
    else:
        train_ds = Database(train_meta, train_nc, nb_data=nb_train, target_size=target)
        val_ds   = Database(val_meta,   val_nc,   nb_data=nb_val,   target_size=target)

    trainloader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        pin_memory=(device.type == "cuda"),
        num_workers=num_workers, persistent_workers=(num_workers > 0),
    )
    val_batch = min(batch_size, len(val_ds)) if len(val_ds) > 0 else 1
    valloader = DataLoader(
        val_ds, batch_size=val_batch, shuffle=False,
        pin_memory=(device.type == "cuda"),
        num_workers=0,
    )
    log.info(f"训练集 {len(train_ds)} 条，验证集 {len(val_ds)} 条，"
             f"steps/epoch={len(trainloader)}")

    # ── 模型 ─────────────────────────────────────────────────────────────────
    model = ACANN_v3(
        nb_gc=nb_gc, n_peaks=n_peaks,
        w1=train_w1, w2=train_w2,
        omega1_range=(o1_start, o1_end),
        omega2_range=(o2_start, o2_end),
        d_hidden=d_hidden, n_layers=n_layers,
        dropout=dropout,
        dw1=dw1, dw2=dw2,
    ).float().to(device)
    log.info(f"ACANN_v3 参数量: {model.count_params():,}")

    ema = EMA(model, decay=ema_decay) if use_ema else None

    # ── 优化器 ───────────────────────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    total_steps = len(trainloader) * epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=total_steps,
        pct_start=warmup_epochs / epochs,
        anneal_strategy="cos",
        final_div_factor=lr / lr_min,
    )

    # ── 断点续训 ─────────────────────────────────────────────────────────────
    start_epoch = 0
    step        = -1
    best_val    = float("inf")

    if resume_path and Path(resume_path).exists():
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt and use_amp:
            scaler.load_state_dict(ckpt["scaler"])
        if ema and "ema_shadow" in ckpt:
            ema.shadow = {k: v.to(device) for k, v in ckpt["ema_shadow"].items()}
        start_epoch = ckpt.get("epoch", 0) + 1
        step        = ckpt.get("step", -1)
        best_val    = ckpt.get("best_val", float("inf"))
        log.info(f"从 {resume_path} 恢复，epoch={start_epoch}, step={step}")

    # ── 历史记录 ─────────────────────────────────────────────────────────────
    step_hist, train_hist, val_hist = [], [], []

    # ── 验证函数 ─────────────────────────────────────────────────────────────
    @torch.no_grad()
    def validate():
        model.eval()
        total, n = 0.0, 0
        ctx = ema.apply() if ema else _null_ctx()
        with ctx:
            for G_v, A_v in valloader:
                G_v = G_v.to(device, non_blocking=True)
                A_v = A_v.to(device, non_blocking=True)
                with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                    A_p = model(G_v)
                loss, _ = combined_loss(A_p.float(), A_v.float(), mse_w, l1_w)
                total += loss.item() * G_v.size(0)
                n     += G_v.size(0)
        model.train()
        return total / max(n, 1)

    # ── 训练循环 ─────────────────────────────────────────────────────────────
    log.info("开始训练 ...")
    t0 = time.time()
    model.train()

    for epoch in range(start_epoch, epochs):
        for G, A in trainloader:
            step += 1
            G = G.to(device, non_blocking=True)
            A = A.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                A_pred, params = model.forward_with_params(G)

            # 损失在 float32 下计算，避免 A 的大值在 float16 下溢出
            loss, sub = combined_loss(A_pred.float(), A.float(), mse_w, l1_w)

            # σ 正则项 + 峰门控稀疏正则
            sig_reg = model.sigma_regularization(params["sigma1"], params["sigma2"])
            gate_mean = params["gate"].mean()
            loss = loss + 0.05 * sig_reg + gate_w * gate_mean
            sub["sig_reg"] = sig_reg.item()
            sub["gate_mean"] = gate_mean.item()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if ema:
                ema.update()

            if step % print_every == 0:
                lr_now = scheduler.get_last_lr()[0]
                log.info(
                    f"Epoch {epoch+1:3d}/{epochs}  step {step:6d}  "
                    f"lr={lr_now:.2e}  train={loss.item():.6f}  "
                    f"[mse_s1={sub['mse_s1']:.6f} l1_s1={sub['l1_s1']:.5f} "
                    f"sig={sub.get('sig_reg',0):.4f} gate={sub.get('gate_mean',0):.3f}]"
                )

            if step % val_every == 0 and step > 0:
                val_loss = validate()
                step_hist.append(step)
                train_hist.append(loss.item())
                val_hist.append(val_loss)
                log.info(f"  ↳ Val loss = {val_loss:.6f}  "
                         f"(best={best_val:.6f}, elapsed={time.time()-t0:.0f}s)")

                if val_loss < best_val:
                    best_val = val_loss
                    _save(ckpt_dir, "best.pth", model, optimizer, scheduler,
                          scaler, ema, epoch, step, best_val)
                    log.info(f"  ★ 保存最优模型 best.pth (val={best_val:.6f})")

        _save(ckpt_dir, "last.pth", model, optimizer, scheduler,
              scaler, ema, epoch, step, best_val)

    log.info(f"训练完成，总用时 {(time.time()-t0)/60:.1f} min，最优验证损失 {best_val:.6f}")

    # ── 绘制损失曲线 ─────────────────────────────────────────────────────────
    if len(step_hist) >= 2:
        w = max(1, len(step_hist) // 20)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.loglog(step_hist, rolling_mean(train_hist, w), label="Train (smoothed)")
        ax.loglog(step_hist, rolling_mean(val_hist,   w), label="Val   (smoothed)")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("ACANN-v3 Training Loss")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, "loss_curve_v3.png"), dpi=150)
        plt.close(fig)
        log.info(f"损失曲线已保存至 {plot_dir}/loss_curve_v3.png")


# ─────────────────────────────────────────────────────────────────────────────
# 辅助工具
# ─────────────────────────────────────────────────────────────────────────────

def _save(ckpt_dir, name, model, optimizer, scheduler, scaler, ema,
          epoch, step, best_val):
    state = {
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler":    scaler.state_dict(),
        "epoch":     epoch,
        "step":      step,
        "best_val":  best_val,
    }
    if ema is not None:
        state["ema_shadow"] = {k: v.cpu() for k, v in ema.shadow.items()}
    torch.save(state, os.path.join(ckpt_dir, name))


class _null_ctx:
    def __enter__(self): return self
    def __exit__(self, *_): pass


# ─────────────────────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default=str(ROOT / "config" / "train_config.ini"),
        help="训练配置文件路径"
    )
    parser.add_argument(
        "--resume", default=None,
        help="断点续训，传入 .pth 文件路径"
    )
    args = parser.parse_args()
    train(args.config, args.resume)
