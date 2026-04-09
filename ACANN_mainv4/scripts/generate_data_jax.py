"""
使用 JAX 生成复频域谱函数 A 和虚时格林函数 G 的训练数据。
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import configparser
import os
import time
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from ACANN_mainv4.data.save_memmap import save_A_memmap


def read_parameters(input_file):
    """读取参数文件"""
    config = configparser.ConfigParser()
    config.read(input_file)
    return config["params"]


def generate_data(input_file):
    start_time = time.time()
    params = read_parameters(input_file)

    BETA = params.getfloat("beta", 80.0)
    NB_DATA = params.getint("nb_data", 1000)
    NB_PICS = params.getint("nb_pics", 21)
    output_prefix = params.get("output", "output")
    NB_GC = params.getint("nb_gc", 64)
    NB_OMEGA1 = params.getint("nb_omega1", 1024)
    NB_OMEGA2 = params.getint("nb_omega2", 1024)
    OMEGA_1 = params.getfloat("omega1", 5.0)
    OMEGA_2 = params.getfloat("omega2", 5.0)
    NB_TAU = params.getint("nb_tau", 153)
    legendre_method = params.get("legendre_method", "exact")
    integral_tol = params.getfloat("integral_tol", 1e-4)
    show_example = params.getboolean("example", False)
    NOISE_LEVEL = params.getfloat("noise_level", 0.001)
    seed = int(start_time)
    output_dir = params.get("output_dir", "data_output")
    tau_grid = params.get("tau_grid", "")

    print(f"使用参数: beta={BETA}, nb_data={NB_DATA}, nb_pics={NB_PICS}")
    print(f"nb_gl={NB_GC}, omega1={OMEGA_1}, omega2={OMEGA_2}")
    print(f", method={legendre_method}, noise={NOISE_LEVEL}")

    key = jax.random.PRNGKey(seed)
    Rmax = 3

    omega1 = jnp.linspace(0.005, OMEGA_1, NB_OMEGA1)
    omega2 = jnp.linspace(0, OMEGA_2, NB_OMEGA2)

    n_active = jnp.full((NB_DATA,), 3)
    key, subkey = jax.random.split(key)
    wr = jax.random.uniform(subkey, (NB_DATA, Rmax), minval=0.0, maxval=2.0)
    wr = jnp.sort(wr, axis=1)

    min_sep = 0.3
    wr = wr.at[:, 1].set(jnp.maximum(wr[:, 1], wr[:, 0] + min_sep))
    wr = wr.at[:, 2].set(jnp.maximum(wr[:, 2], wr[:, 1] + min_sep))
    wr = jnp.clip(wr, 0.0, 2.0)

    key, subkey = jax.random.split(key)
    sigma1 = jax.random.uniform(subkey, (NB_DATA, Rmax), minval=0.02, maxval=0.06)
    key, subkey = jax.random.split(key)
    sigma2 = jax.random.uniform(subkey, (NB_DATA, Rmax), minval=0.003, maxval=0.01)
    key, subkey = jax.random.split(key)
    amp = jax.random.uniform(subkey, (NB_DATA, Rmax), minval=0.5, maxval=2.0)

    peak_ids = jnp.arange(Rmax)[None, :]
    mask = (peak_ids < n_active[:, None]).astype(jnp.float32)
    amp = amp * mask
    wi = jnp.zeros((NB_DATA, Rmax))

    def compute_A(i):
        W1, W2 = jnp.meshgrid(omega1, omega2, indexing="ij")

        def add_peak(j, val):
            mu1 = wr[i, j]
            mu2 = wi[i, j]
            s1 = sigma1[i, j]
            s2 = sigma2[i, j]
            a = amp[i, j]
            gauss = a * jnp.exp(
                -(W1 - mu1) ** 2 / (2 * s1 ** 2)
                - (W2 - mu2) ** 2 / (2 * s2 ** 2)
            ) / (2 * jnp.pi * s1 * s2)
            return val + gauss

        A_i = jax.lax.fori_loop(0, Rmax, add_peak, jnp.zeros_like(W1))
        dw1 = omega1[1] - omega1[0]
        dw2 = omega2[1] - omega2[0]
        norm = jnp.sum(A_i) * dw1 * dw2
        return A_i / norm

    A = jax.vmap(compute_A)(jnp.arange(NB_DATA))
    print(A.shape)
    print("A max =", jnp.max(A).item())
    print("A min =", jnp.min(A).item())

    idx = 0
    A0 = np.array(A[idx])
    w1 = np.array(omega1)
    w2 = np.array(omega2)
    plt.figure(figsize=(6, 4))
    plt.imshow(A0.T, origin="lower", extent=[w1[0], w1[-1], w2[0], w2[-1]], aspect="auto")
    plt.xlabel(r"$\omega_1$")
    plt.ylabel(r"$\omega_2$")
    plt.title(fr"$A(\omega_1,\omega_2)$ for sample {idx}")
    plt.colorbar(label="A")
    plt.tight_layout()
    plt.show()

    print("green function")
    file_path = tau_grid.strip() if tau_grid else r"D:\data\Jz=0\tgrid.dat"
    if not Path(file_path).exists():
        raise FileNotFoundError(
            f"tau 网格文件不存在: {file_path}。请在 config/params.ini 的 [params] 中设置 tau_grid=你的tgrid.dat路径"
        )
    tau = np.loadtxt(file_path).flatten()
    center = 40.0
    tau_left = tau
    tau_right = 2 * center - tau_left
    tau_right = tau_right[::-1]
    tau_right = tau_right[tau_right != center]
    tau = np.concatenate([tau_left, tau_right])
    print("tau shape =", tau.shape)

    G = jnp.zeros((NB_DATA, NB_TAU))

    @jax.jit
    def kernel_2d(tau_val, omega1_val, omega2_val):
        num = (
            2 * jnp.exp(-(BETA - tau_val) * omega1_val) * jnp.cos(-(BETA - tau_val) * omega2_val)
            + 2 * jnp.exp(-tau_val * omega1_val) * jnp.cos(-tau_val * omega2_val)
        )
        den = 2 * (1.0 + jnp.exp(-BETA * omega1_val) * jnp.cos(-BETA * omega2_val))
        return num / den

    vec_kernel_2d = jax.vmap(
        jax.vmap(
            jax.vmap(kernel_2d, in_axes=(None, None, 0)),
            in_axes=(None, 0, None)
        ),
        in_axes=(0, None, None)
    )

    def trapz_weights(x):
        dx = jnp.diff(x)
        w = jnp.zeros_like(x)
        w = w.at[0].set(dx[0] / 2.0)
        w = w.at[-1].set(dx[-1] / 2.0)
        w = w.at[1:-1].set((x[2:] - x[:-2]) / 2.0)
        return w

    omega1_ = omega1.astype(jnp.float64)
    omega2_ = omega2.astype(jnp.float64)
    tau_ = tau.astype(jnp.float64)
    w1 = trapz_weights(omega1_)
    w2 = trapz_weights(omega2_)
    Kbar = vec_kernel_2d(tau_, omega1_, omega2_).astype(jnp.float64)

    def compute_G_all(A):
        A = A.astype(jnp.float64)
        tmp = jnp.einsum("nij,tij,j->nti", A, Kbar, w2)
        G = jnp.einsum("nti,i->nt", tmp, w1)
        return G

    G = compute_G_all(A)
    print("G shape:", G.shape)

    if NOISE_LEVEL > 0:
        master = jax.random.PRNGKey(0)
        keys = jax.random.split(master, G.shape[0])

        def add_noise_one_sample(g_n, k):
            return g_n + jax.random.normal(k, shape=g_n.shape, dtype=g_n.dtype) * NOISE_LEVEL

        G = jax.vmap(add_noise_one_sample, in_axes=(0, 0))(G, keys)

    print("使用近似方法计算Chebyshev系数")
    u = 2.0 * tau / BETA - 1.0
    u = u.astype(jnp.float64)

    def cheb_T_matrix(u, NB_GL):
        Phi = jnp.zeros((u.shape[0], NB_GL), dtype=jnp.float64)
        Phi = Phi.at[:, 0].set(1.0)
        if NB_GL > 1:
            Phi = Phi.at[:, 1].set(u)
        for l in range(2, NB_GL):
            Phi = Phi.at[:, l].set(2.0 * u * Phi[:, l - 1] - Phi[:, l - 2])
        return Phi

    Phi = cheb_T_matrix(u, NB_GC)
    Phi_pinv = jnp.linalg.pinv(Phi)
    nc = (Phi_pinv @ G.T).T
    print("G_c shape", nc.shape)
    print(f"计算Chebyshev系数完成, 总用时: {time.time() - start_time:.2f}秒")

    out_path = ROOT / output_dir
    os.makedirs(out_path, exist_ok=True)

    with open(out_path / f"G_{output_prefix}.csv", "ab") as f:
        np.savetxt(f, np.array(G), delimiter=",")

    with open(out_path / f"nl_{output_prefix}_{NOISE_LEVEL}.csv", "ab") as f:
        np.savetxt(f, np.array(nc), delimiter=",")

    save_A_memmap(
        A_batch=np.array(A),
        omega1_grid=np.array(omega1),
        omega2_grid=np.array(omega2),
        base_dir=str(out_path),
        name=f"A_{output_prefix}",
        N_total=100000
    )

    if show_example:
        plot_example(omega1, A[0], tau, G[0], Phi, nc[0], NB_GC, BETA)


def plot_example(omega, A, tau, G, Phi, nc, NB_GL, BETA):
    """绘制示例图形"""
    plt.figure(figsize=(15, 6))
    plt.plot(tau, G)
    plt.grid()
    plt.show()

    u = 2.0 * tau / BETA - 1.0
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
    jax.config.update("jax_enable_x64", True)
    config_path = ROOT / "config" / "params.ini"
    generate_data(str(config_path))
