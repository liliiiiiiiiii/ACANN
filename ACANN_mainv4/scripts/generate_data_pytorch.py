"""
使用 PyTorch 生成复频域谱函数 A 和虚时格林函数 G 的训练数据。
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import configparser
import math
import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt

from ACANN_mainv4.data.save_memmap import save_A_memmap


def _get_device():
    """获取可用设备。若 CUDA 不可用或设置 FORCE_CPU=1 则使用 CPU。"""
    if os.environ.get("FORCE_CPU", "0").strip().lower() in ("1", "true", "yes"):
        print("FORCE_CPU=1, 使用 CPU")
        return torch.device("cpu")
    if not torch.cuda.is_available():
        print("使用 CPU")
        return torch.device("cpu")
    try:
        # 测试稍大张量，避免小测试通过但大分配失败
        x = torch.zeros(1000, 1000, device="cuda", dtype=torch.float64)
        del x
        torch.cuda.empty_cache()
        print("使用 CUDA")
        return torch.device("cuda")
    except RuntimeError as e:
        print(f"CUDA 不可用 ({e}), 回退到 CPU")
        return torch.device("cpu")


def read_parameters(input_file):
    """读取参数文件"""
    config = configparser.ConfigParser()
    config.read(input_file, encoding="utf-8")
    return config["params"]


def generate_data(input_file):
    start_time = time.time()
    params = read_parameters(input_file)

    BETA          = params.getfloat("beta", 80.0)
    NB_DATA       = params.getint("nb_data", 1000)
    NB_PICS       = params.getint("nb_pics", 21)
    output_prefix = params.get("output", "output")
    NB_GC         = params.getint("nb_gc", 128)
    NB_OMEGA1     = params.getint("nb_omega1", 500)
    NB_OMEGA2     = params.getint("nb_omega2", 500)
    OMEGA_1       = params.getfloat("omega1", 4.995)
    OMEGA_2       = params.getfloat("omega2", 4.995)
    NB_TAU        = params.getint("nb_tau", 152)
    legendre_method = params.get("legendre_method", "exact")
    integral_tol  = params.getfloat("integral_tol", 1e-4)
    show_example  = params.getboolean("example", False)
    NOISE_LEVEL   = params.getfloat("noise_level", 0.001)
    seed          = int(start_time)
    output_dir    = params.get("output_dir", "data_output")
    tau_grid      = params.get("tau_grid", "")
    # 每批在 GPU 上处理的样本数，越小显存占用越低
    BATCH_SIZE    = params.getint("batch_size", 200)
    # memmap 预分配总条数；磁盘占用 = N_TOTAL × W1 × W2 × 4 字节
    N_TOTAL       = params.getint("n_total", NB_DATA)

    print(f"使用参数: beta={BETA}, nb_data={NB_DATA}, nb_pics={NB_PICS}")
    print(f"nb_gl={NB_GC}, omega1={OMEGA_1}, omega2={OMEGA_2}")
    print(f", method={legendre_method}, noise={NOISE_LEVEL}")
    print(f"batch_size={BATCH_SIZE}  (显存预算: Kbar≈"
          f"{NB_TAU*NB_OMEGA1*NB_OMEGA2*8/1e9:.2f}GB + "
          f"A_batch≈{BATCH_SIZE*NB_OMEGA1*NB_OMEGA2*8/1e9:.2f}GB)")

    torch.manual_seed(seed)
    device = _get_device()

    Rmax = 3
    omega1 = torch.linspace(0.005, OMEGA_1, NB_OMEGA1, dtype=torch.float64, device=device)
    omega2 = torch.linspace(0.005, OMEGA_2, NB_OMEGA2, dtype=torch.float64, device=device)

    # -------- 读取 tau 网格（与批次无关，预先加载） --------
    file_path = tau_grid.strip() if tau_grid else r"D:\dev\ACANN\tgrid.dat"
    if not Path(file_path).exists():
        raise FileNotFoundError(
            f"tau 网格文件不存在: {file_path}。\n"
            "请在 config/params.ini 的 [params] 中设置 tau_grid=你的tgrid.dat路径"
        )
    tau_np = np.loadtxt(file_path).flatten()
    center    = 40.0
    tau_left  = tau_np
    tau_right = 2 * center - tau_left
    tau_right = tau_right[::-1]
    tau_right = tau_right[tau_right != center]
    tau_np    = np.concatenate([tau_left, tau_right])
    print("tau shape =", tau_np.shape)

    tau = torch.tensor(tau_np, dtype=torch.float64, device=device)

    # -------- 预计算 Kbar (T, W1, W2) ——只算一次，各批复用 --------
    print("预计算 Kbar ...")
    tau_e = tau[:, None, None]
    o1_e  = omega1[None, :, None]
    o2_e  = omega2[None, None, :]
    num   = (
        2 * torch.exp(-(BETA - tau_e) * o1_e) * torch.cos(-(BETA - tau_e) * o2_e)
        + 2 * torch.exp(-tau_e * o1_e)         * torch.cos(-tau_e * o2_e)
    )
    den   = 2 * (1.0 + torch.exp(-BETA * o1_e) * torch.cos(-BETA * o2_e))
    Kbar  = num / den   # (T, W1, W2)
    del tau_e, o1_e, o2_e, num, den
    torch.cuda.empty_cache() if device.type == "cuda" else None

    # -------- 梯形权重 --------
    def trapz_weights(x: torch.Tensor) -> torch.Tensor:
        dx = x[1:] - x[:-1]
        w  = torch.zeros_like(x)
        w[0]    = dx[0] / 2.0
        w[-1]   = dx[-1] / 2.0
        w[1:-1] = (x[2:] - x[:-2]) / 2.0
        return w

    w1_weights = trapz_weights(omega1)   # (W1,)
    w2_weights = trapz_weights(omega2)   # (W2,)

    # -------- 预计算 Chebyshev 伪逆——只算一次 --------
    u = 2.0 * tau / BETA - 1.0

    def cheb_T_matrix(u: torch.Tensor, NB_GL: int) -> torch.Tensor:
        T_len = u.shape[0]
        Phi   = torch.zeros((T_len, NB_GL), dtype=torch.float64, device=device)
        Phi[:, 0] = 1.0
        if NB_GL > 1:
            Phi[:, 1] = u
        for l in range(2, NB_GL):
            Phi[:, l] = 2.0 * u * Phi[:, l - 1] - Phi[:, l - 2]
        return Phi

    Phi      = cheb_T_matrix(u, NB_GC)      # (NB_TAU, NB_GC)
    Phi_pinv = torch.linalg.pinv(Phi)        # (NB_GC, NB_TAU)

    # -------- 准备输出目录 / 文件句柄 --------
    out_path = ROOT / output_dir
    os.makedirs(out_path, exist_ok=True)
    f_G  = open(out_path / f"G_{output_prefix}.csv",                  "ab")
    f_nc = open(out_path / f"nl_{output_prefix}_{NOISE_LEVEL}.csv",   "ab")

    # 用于 show_example 时保存第一批的可视化数据
    first_batch_data = {}

    # -------- 批量生成 --------
    W1, W2  = torch.meshgrid(omega1, omega2, indexing="ij")   # (W1, W2) — 各批共用
    W1_grid = W1[None, :, :]   # (1, W1, W2)
    W2_grid = W2[None, :, :]
    dw1 = omega1[1] - omega1[0]
    dw2 = omega2[1] - omega2[0]
    Kbar_flat = Kbar.reshape(Kbar.shape[0], -1)   # (T, W1*W2) — view，无额外显存

    n_batches = math.ceil(NB_DATA / BATCH_SIZE)
    print(f"共 {NB_DATA} 个样本，分 {n_batches} 批（每批 {BATCH_SIZE}）处理")

    for batch_idx in range(n_batches):
        b_start = batch_idx * BATCH_SIZE
        b       = min(BATCH_SIZE, NB_DATA - b_start)

        # -- 生成本批随机参数 --
        n_active = torch.randint(1, 4, (b,), dtype=torch.long, device=device)
        wr = torch.rand(b, Rmax, dtype=torch.float64, device=device) * 2.2

        sigma1 = torch.rand(b, Rmax, dtype=torch.float64, device=device) * (0.3 - 0.02) + 0.02
        sigma2 = torch.rand(b, Rmax, dtype=torch.float64, device=device) * (0.01 - 0.003) + 0.003
        amp    = torch.rand(b, Rmax, dtype=torch.float64, device=device) * (2.0 - 0.5) + 0.5

        peak_ids = torch.arange(Rmax, device=device)[None, :]
        amp     *= (peak_ids < n_active[:, None]).to(dtype=torch.float64)
        wi       = torch.zeros(b, Rmax, dtype=torch.float64, device=device)

        # -- 计算谱函数 A_batch (b, W1, W2) --
        A_b = torch.zeros(b, NB_OMEGA1, NB_OMEGA2, dtype=torch.float64, device=device)
        for j in range(Rmax):
            mu1 = wr[:, j, None, None]
            mu2 = wi[:, j, None, None]
            s1  = sigma1[:, j, None, None]
            s2  = sigma2[:, j, None, None]
            a_j = amp[:, j, None, None]
            A_b += a_j * torch.exp(
                -(W1_grid - mu1) ** 2 / (2 * s1 ** 2)
                - (W2_grid - mu2) ** 2 / (2 * s2 ** 2)
            ) / (2 * math.pi * s1 * s2)

        norm_factor = A_b.sum(dim=(-2, -1)) * dw1 * dw2
        A_b /= norm_factor[:, None, None]

        # 先把 A 搬到 CPU（后续 save_A_memmap 用），再原地乘权重算 G，节省一份显存
        A_np = A_b.cpu().numpy()

        # -- 计算 G_batch (b, T)：G = A_w.reshape(b,-1) @ Kbar_flat.T --
        A_b *= w1_weights[None, :, None]
        A_b *= w2_weights[None, None, :]
        G_b  = A_b.reshape(b, -1) @ Kbar_flat.T   # (b, T)
        del A_b
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if NOISE_LEVEL > 0:
            G_b = G_b + torch.randn_like(G_b) * NOISE_LEVEL

        # -- 计算 Chebyshev 系数 nc_batch (b, NB_GC) --
        nc_b = (Phi_pinv @ G_b.T).T   # (b, NB_GC)

        # -- 保存本批 --
        G_np  = G_b.cpu().numpy()
        nc_np = nc_b.cpu().numpy()
        np.savetxt(f_G,  G_np,  delimiter=",")
        np.savetxt(f_nc, nc_np, delimiter=",")
        save_A_memmap(
            A_batch=A_np,
            omega1_grid=omega1.cpu().numpy() if batch_idx == 0 else None,
            omega2_grid=omega2.cpu().numpy() if batch_idx == 0 else None,
            base_dir=str(out_path),
            name=f"A_{output_prefix}",
            N_total=N_TOTAL
        )

        # 保留第一批供可视化
        if batch_idx == 0:
            first_batch_data = {
                "A0":   A_np[0],
                "G0":   G_np[0],
                "nc0":  nc_np[0],
            }
            print(f"  [batch 0] A max={A_np.max():.4f}, A min={A_np.min():.6f}")
            print(f"  [batch 0] G shape: {G_np.shape}")

        del G_b, nc_b, G_np, nc_np, A_np
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if (batch_idx + 1) % max(1, n_batches // 10) == 0 or batch_idx == n_batches - 1:
            pct = (batch_idx + 1) / n_batches * 100
            print(f"  进度: {batch_idx+1}/{n_batches} ({pct:.0f}%)  "
                  f"已用时 {time.time()-start_time:.1f}s")

    f_G.close()
    f_nc.close()
    print(f"全部完成，总用时: {time.time() - start_time:.2f}秒")

    # -------- 可视化（第一批第一个样本） --------
    w1_np = omega1.cpu().numpy()
    w2_np = omega2.cpu().numpy()
    plt.figure(figsize=(6, 4))
    plt.imshow(
        first_batch_data["A0"].T,
        origin="lower",
        extent=[w1_np[0], w1_np[-1], w2_np[0], w2_np[-1]],
        aspect="auto"
    )
    plt.xlabel(r"$\omega_1$")
    plt.ylabel(r"$\omega_2$")
    plt.title(r"$A(\omega_1,\omega_2)$ for sample 0")
    plt.colorbar(label="A")
    plt.tight_layout()
    plt.show()

    if show_example:
        plot_example(
            omega1.cpu().numpy(), first_batch_data["A0"],
            tau_np, first_batch_data["G0"],
            Phi.cpu().numpy(), first_batch_data["nc0"],
            NB_GC, BETA
        )


def plot_example(omega, A, tau, G, Phi, nc, NB_GL, BETA):
    """绘制示例图形"""
    plt.figure(figsize=(15, 6))
    plt.plot(tau, G)
    plt.grid()
    plt.show()

    G_recon = nc @ Phi.T
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.array(tau), np.array(G), "b-", label="origin")
    ax.plot(np.array(tau), np.array(G_recon), "r--", label="Chebyshev reconstruct")
    ax.grid(True)
    ax.legend()
    ax_inset = fig.add_axes([0.62, 0.22, 0.30, 0.30])
    ax_inset.plot(np.arange(NB_GL), np.array(nc[:NB_GL]), "x-")
    ax_inset.set_xlabel("l", fontsize=8)
    ax_inset.set_ylabel("coef", fontsize=8)
    ax_inset.set_title("Chebyshev coeffs", fontsize=9)
    ax_inset.grid(True)
    ax_inset.tick_params(labelsize=8)
    plt.show()


if __name__ == "__main__":
    config_path = ROOT / "config" / "params.ini"
    generate_data(str(config_path))
