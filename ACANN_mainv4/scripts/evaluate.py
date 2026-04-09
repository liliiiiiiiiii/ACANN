"""
ACANN_v3 评估脚本

读取训练保存的 .pth checkpoint，从验证集中随机抽取样本，
绘制预测谱函数与真实谱函数的多组对比图，输出预测的高斯参数，
直观展示模型效果。

运行示例
--------
  python scripts/evaluate.py
  python scripts/evaluate.py --ckpt checkpoints_2.0/best.pth --n 8
  python scripts/evaluate.py --ckpt checkpoints/best.pth --config config/train_config.ini --n 6 --out plots/eval
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import argparse
import configparser
import os
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from ACANN_mainv4.models.acann_v2 import ACANN_v3
from ACANN_mainv4.data.dataset import Database


# ─────────────────────────────────────────────────────────────────────────────
# 配置读取
# ─────────────────────────────────────────────────────────────────────────────

def read_config(path: str) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    cfg.read(path, encoding="utf-8")
    return cfg


def _read_omega_and_dw(cfg, train_w1, train_w2):
    """从 meta 读取 omega 范围和 dw。"""
    meta_path = ROOT / cfg.get("data", "train_meta")
    o1_start, o1_end = 0.005, 4.995
    o2_start, o2_end = 0.005,   4.995
    dw1, dw2 = 5.0 / train_w1, 5.0 / train_w2

    if meta_path.exists():
        meta = np.load(str(meta_path), allow_pickle=True)
        if "omega1" in meta.files and meta["omega1"] is not None:
            o1 = meta["omega1"].astype(np.float64)
            if len(o1) > 1:
                o1_start, o1_end = float(o1[0]), float(o1[-1])
                dw1 = float(o1[-1] - o1[0]) / max(train_w1 - 1, 1)
        if "omega2" in meta.files and meta["omega2"] is not None:
            o2 = meta["omega2"].astype(np.float64)
            if len(o2) > 1:
                o2_start, o2_end = float(o2[0]), float(o2[-1])
                dw2 = float(o2[-1] - o2[0]) / max(train_w2 - 1, 1)

    return o1_start, o1_end, o2_start, o2_end, dw1, dw2


# ─────────────────────────────────────────────────────────────────────────────
# 加载模型
# ─────────────────────────────────────────────────────────────────────────────

def load_model(ckpt_path: str, cfg: configparser.ConfigParser,
               device: torch.device) -> torch.nn.Module:
    nb_gc    = cfg.getint  ("model", "nb_gc",    fallback=64)
    n_peaks  = cfg.getint  ("model", "n_peaks",  fallback=3)
    d_hidden = cfg.getint  ("model", "d_hidden", fallback=512)
    n_layers = cfg.getint  ("model", "n_layers", fallback=6)
    dropout  = cfg.getfloat("model", "dropout",  fallback=0.05)
    train_w1 = cfg.getint  ("data",  "train_w1", fallback=1024)
    train_w2 = cfg.getint  ("data",  "train_w2", fallback=1024)

    o1s, o1e, o2s, o2e, dw1, dw2 = _read_omega_and_dw(cfg, train_w1, train_w2)

    model = ACANN_v3(
        nb_gc=nb_gc, n_peaks=n_peaks,
        w1=train_w1, w2=train_w2,
        omega1_range=(o1s, o1e), omega2_range=(o2s, o2e),
        d_hidden=d_hidden, n_layers=n_layers, dropout=dropout,
        dw1=dw1, dw2=dw2,
    ).float().to(device)

    ckpt = torch.load(ckpt_path, map_location=device)

    if "ema_shadow" in ckpt:
        state = model.state_dict()
        for k, v in ckpt["ema_shadow"].items():
            if k in state:
                state[k] = v.to(device)
        model.load_state_dict(state)
        print("Loaded EMA weights")
    else:
        model.load_state_dict(ckpt["model"])
        print("Loaded model weights")

    epoch    = ckpt.get("epoch",    "?")
    best_val = ckpt.get("best_val", float("nan"))
    print(f"Checkpoint: epoch={epoch}, best_val_loss={best_val:.6f}")

    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 加载数据集
# ─────────────────────────────────────────────────────────────────────────────

def load_val_dataset(cfg: configparser.ConfigParser):
    train_meta = str(ROOT / cfg.get("data", "train_meta"))
    train_nc   = str(ROOT / cfg.get("data", "train_nc"))
    val_meta   = str(ROOT / cfg.get("data", "val_meta"))
    val_nc     = str(ROOT / cfg.get("data", "val_nc"))
    nb_train   = cfg.getint("data", "nb_train", fallback=5000)
    nb_val     = cfg.getint("data", "nb_val",   fallback=500)
    train_w1   = cfg.getint("data", "train_w1", fallback=1024)
    train_w2   = cfg.getint("data", "train_w2", fallback=1024)
    target     = (train_w1, train_w2)

    same_dataset = (train_meta == val_meta and train_nc == val_nc)

    if same_dataset:
        full_n  = nb_train + nb_val
        full_ds = Database(train_meta, train_nc, nb_data=full_n, target_size=target)
        actual_val = min(nb_val, len(full_ds) - nb_train)
        if actual_val <= 0:
            print("Warning: val set is empty, falling back to first samples.")
            return Database(train_meta, train_nc, nb_data=min(nb_val, 500),
                            target_size=target)
        _, val_ds = random_split(
            full_ds, [nb_train, actual_val],
            generator=torch.Generator().manual_seed(42),
        )
    else:
        val_ds = Database(val_meta, val_nc, nb_data=nb_val, target_size=target)

    return val_ds


def load_omega_grids(cfg: configparser.ConfigParser):
    meta_path = ROOT / cfg.get("data", "train_meta")
    train_w1  = cfg.getint("data", "train_w1", fallback=1024)
    train_w2  = cfg.getint("data", "train_w2", fallback=1024)
    o1_start, o1_end = 0.005, 4.995
    o2_start, o2_end =0.005, 4.995

    if meta_path.exists():
        meta = np.load(str(meta_path), allow_pickle=True)
        if "omega1" in meta.files and meta["omega1"] is not None:
            o1 = meta["omega1"]
            if o1.ndim == 1 and len(o1) > 1:
                o1_start, o1_end = float(o1[0]), float(o1[-1])
        if "omega2" in meta.files and meta["omega2"] is not None:
            o2 = meta["omega2"]
            if o2.ndim == 1 and len(o2) > 1:
                o2_start, o2_end = float(o2[0]), float(o2[-1])

    omega1 = np.linspace(o1_start, o1_end, train_w1, dtype=np.float32)
    omega2 = np.linspace(o2_start, o2_end, train_w2, dtype=np.float32)
    return omega1, omega2


# ─────────────────────────────────────────────────────────────────────────────
# 误差指标
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(A_pred: np.ndarray, A_true: np.ndarray,
                    dw1: float, dw2: float) -> dict:
    eps = 1e-8
    p   = A_pred * dw1 * dw2
    q   = A_true * dw1 * dw2

    mask = q > 0
    kl   = np.sum(q[mask] * np.log((q[mask] + eps) / (p[mask] + eps))) if mask.any() else float("nan")

    l1   = np.mean(np.abs(A_pred - A_true))
    rmse = np.sqrt(np.mean((A_pred - A_true) ** 2))
    rel  = np.sum(np.abs(A_pred - A_true)) / (np.sum(np.abs(A_true)) + eps)

    pred_peak = np.unravel_index(np.argmax(A_pred), A_pred.shape)
    true_peak = np.unravel_index(np.argmax(A_true), A_true.shape)
    peak_err  = np.sqrt((pred_peak[0] - true_peak[0])**2 +
                        (pred_peak[1] - true_peak[1])**2)

    return {"KL": kl, "L1": l1, "RMSE": rmse, "RelErr": rel, "PeakErr_px": peak_err}


# ─────────────────────────────────────────────────────────────────────────────
# 绘图
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparisons(
    G_batch: torch.Tensor,
    A_true_batch: torch.Tensor,
    A_pred_batch: torch.Tensor,
    dw1: float,
    dw2: float,
    out_dir: str,
    tag: str = "eval",
    omega1: Optional[np.ndarray] = None,
    omega2: Optional[np.ndarray] = None,
    params_list: Optional[list] = None,
):
    os.makedirs(out_dir, exist_ok=True)
    n = A_true_batch.shape[0]
    all_metrics = []

    w1_axis = omega1 if omega1 is not None and len(omega1) > 1 else np.arange(A_true_batch.shape[1], dtype=np.float32)
    w2_axis = omega2 if omega2 is not None and len(omega2) > 1 else np.arange(A_true_batch.shape[2], dtype=np.float32)

    for i in range(n):
        all_metrics.append(
            compute_metrics(
                A_pred_batch[i].cpu().numpy(),
                A_true_batch[i].cpu().numpy(),
                dw1, dw2,
            )
        )

    # ── 每个样本一行，4 列：ω₁截面 / ω₂截面 / 残差 / 参数标注 ────────────
    rows_per_page = 4
    pages = (n + rows_per_page - 1) // rows_per_page

    for page in range(pages):
        start = page * rows_per_page
        end   = min(start + rows_per_page, n)
        nrows = end - start

        fig, axes = plt.subplots(nrows, 3, figsize=(15, 4.2 * nrows), squeeze=False)
        fig.subplots_adjust(hspace=0.55, wspace=0.32)

        for row, idx in enumerate(range(start, end)):
            A_t = A_true_batch[idx].cpu().numpy()
            A_p = A_pred_batch[idx].cpu().numpy()
            m   = all_metrics[idx]

            true_peak = np.unravel_index(np.argmax(A_t), A_t.shape)
            i1_peak, i2_peak = true_peak

            # ── 列1: 沿 ω₁ 截面 ──────────────────────────────────────────
            ax = axes[row, 0]
            ax.plot(w1_axis, A_t[:, i2_peak], "b-",  linewidth=1.5, label="True")
            ax.plot(w1_axis, A_p[:, i2_peak], "r--", linewidth=1.5, label="Pred")
            ax.fill_between(w1_axis, A_t[:, i2_peak], A_p[:, i2_peak],
                            alpha=0.15, color="red")
            ax.set_xlabel(r"$\omega_1$")
            ax.set_ylabel(r"$A(\omega_1,\omega_2^*)$")
            title_str = (
                fr"Sample {idx}   $\omega_2^*={w2_axis[i2_peak]:.3f}$"
                f"\nL1={m['L1']:.5f}  RelErr={m['RelErr']:.4f}"
            )
            if params_list and idx < len(params_list):
                p = params_list[idx]
                mu1_str = ", ".join(f"{v:.3f}" for v in p["mu1"])
                title_str += f"\nPred peaks: [{mu1_str}]"
            ax.set_title(title_str, fontsize=8)
            ax.legend(fontsize=7)
            ax.grid(True, linestyle="--", alpha=0.4)

            # ── 列2: 沿 ω₂ 截面 ──────────────────────────────────────────
            ax = axes[row, 1]
            ax.plot(w2_axis, A_t[i1_peak, :], "b-",  linewidth=1.5, label="True")
            ax.plot(w2_axis, A_p[i1_peak, :], "r--", linewidth=1.5, label="Pred")
            ax.fill_between(w2_axis, A_t[i1_peak, :], A_p[i1_peak, :],
                            alpha=0.15, color="red")
            ax.set_xlabel(r"$\omega_2$")
            ax.set_ylabel(r"$A(\omega_1^*,\omega_2)$")
            ax.set_title(
                fr"Sample {idx}   $\omega_1^*={w1_axis[i1_peak]:.3f}$"
                f"\nKL={m['KL']:.4f}  RMSE={m['RMSE']:.5f}",
                fontsize=8,
            )
            ax.legend(fontsize=7)
            ax.grid(True, linestyle="--", alpha=0.4)

            # ── 列3: 残差 ────────────────────────────────────────────────
            diff_slice = A_p[:, i2_peak] - A_t[:, i2_peak]
            ax = axes[row, 2]
            ax.plot(w1_axis, diff_slice, color="green", linewidth=1.2)
            ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
            ax.fill_between(w1_axis, diff_slice, 0,
                            where=(diff_slice > 0), alpha=0.25, color="red",
                            label="over-pred")
            ax.fill_between(w1_axis, diff_slice, 0,
                            where=(diff_slice < 0), alpha=0.25, color="blue",
                            label="under-pred")
            ax.set_xlabel(r"$\omega_1$")
            ax.set_ylabel(r"$\hat{A} - A$")
            ax.set_title(
                fr"Sample {idx}   residual, PeakErr={m['PeakErr_px']:.1f}px",
                fontsize=8,
            )
            ax.legend(fontsize=7)
            ax.grid(True, linestyle="--", alpha=0.4)

        save_path = os.path.join(out_dir, f"{tag}_curves_page{page+1}.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {save_path}")

    # ── 误差分布直方图 ───────────────────────────────────────────────────────
    keys   = ["KL", "L1", "RMSE", "RelErr", "PeakErr_px"]
    labels = ["KL divergence", "L1 error", "RMSE", "Relative error", "Peak err (px)"]
    vals   = {k: [m[k] for m in all_metrics] for k in keys}

    fig, axes_stat = plt.subplots(1, len(keys), figsize=(4 * len(keys), 4))
    for ax, k, lab in zip(axes_stat, keys, labels):
        v = np.array(vals[k])
        v = v[np.isfinite(v)]
        if len(v) == 0:
            continue
        ax.hist(v, bins=max(5, n // 3), edgecolor="black", color="steelblue")
        ax.axvline(np.mean(v), color="red", linestyle="--",
                   label=f"mean={np.mean(v):.5f}")
        ax.axvline(np.median(v), color="orange", linestyle=":",
                   label=f"median={np.median(v):.5f}")
        ax.set_title(lab, fontsize=10)
        ax.legend(fontsize=8)

    fig.suptitle(f"Error distribution ({n} samples)", fontsize=13)
    fig.tight_layout()
    stats_path = os.path.join(out_dir, f"{tag}_metrics_hist.png")
    fig.savefig(stats_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {stats_path}")

    print("\n──── Error summary ────")
    for k, lab in zip(keys, labels):
        v = np.array(vals[k])
        v = v[np.isfinite(v)]
        if len(v) == 0:
            print(f"  {lab:22s}  (no finite values)")
            continue
        print(f"  {lab:22s}  mean={np.mean(v):.6f}  median={np.median(v):.6f}"
              f"  std={np.std(v):.6f}")
    print("───────────────────────\n")


# ─────────────────────────────────────────────────────────────────────────────
# 参数表打印
# ─────────────────────────────────────────────────────────────────────────────

def print_params_table(params_list: list, n_show: int = 8):
    """打印预测的高斯参数表。"""
    print("\n──── Predicted Gaussian Parameters ────")
    has_gate = "gate" in params_list[0] if params_list else False
    header = (f"{'Sample':>6}  {'Peak':>4}  {'mu1':>8}  {'mu2':>8}  "
              f"{'sigma1':>8}  {'sigma2':>8}  {'amp':>8}")
    if has_gate:
        header += f"  {'gate':>6}"
    print(header)
    print("─" * (62 + (8 if has_gate else 0)))
    for i, p in enumerate(params_list[:n_show]):
        for k in range(len(p["mu1"])):
            line = (f"{i:6d}  {k+1:4d}  {p['mu1'][k]:8.4f}  {p['mu2'][k]:8.5f}  "
                    f"{p['sigma1'][k]:8.5f}  {p['sigma2'][k]:8.6f}  {p['amp'][k]:8.4f}")
            if has_gate:
                line += f"  {p['gate'][k]:6.3f}"
            print(line)
        if i < min(n_show, len(params_list)) - 1:
            print()
    print("────────────────────────────────────────\n")


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    ckpt_path: str,
    config_path: str,
    n_samples: int = 8,
    out_dir: Optional[str] = None,
    seed: int = 42,
):
    cfg    = read_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model(ckpt_path, cfg, device)

    train_w1 = cfg.getint("data", "train_w1", fallback=1024)
    train_w2 = cfg.getint("data", "train_w2", fallback=1024)
    _, _, _, _, dw1, dw2 = _read_omega_and_dw(cfg, train_w1, train_w2)
    print(f"Grid spacing (for {train_w1}x{train_w2}): dw1={dw1:.6f}  dw2={dw2:.6f}")

    print("Loading validation dataset ...")
    val_ds   = load_val_dataset(cfg)
    n_actual = min(n_samples, len(val_ds))
    print(f"Val set size: {len(val_ds)},  evaluating {n_actual} samples")

    rng  = torch.Generator().manual_seed(seed)
    idxs = torch.randperm(len(val_ds), generator=rng)[:n_actual].tolist()

    G_list, A_list = [], []
    for i in idxs:
        g, a = val_ds[i]
        G_list.append(g)
        A_list.append(a)

    G_batch      = torch.stack(G_list).to(device)
    A_true_batch = torch.stack(A_list).to(device)

    print("Running inference ...")
    with torch.no_grad():
        A_pred_batch, raw_params = model.forward_with_params(G_batch)

    # 整理参数到 CPU list
    params_list = []
    for i in range(n_actual):
        params_list.append({
            "mu1":    raw_params["mu1"][i].cpu().numpy(),
            "mu2":    raw_params["mu2"][i].cpu().numpy(),
            "sigma1": raw_params["sigma1"][i].cpu().numpy(),
            "sigma2": raw_params["sigma2"][i].cpu().numpy(),
            "amp":    raw_params["amp"][i].cpu().numpy(),
            "gate":   raw_params["gate"][i].cpu().numpy(),
        })

    print_params_table(params_list, n_show=n_actual)

    omega1, omega2 = load_omega_grids(cfg)
    print(f"omega1: [{omega1[0]:.3f}, {omega1[-1]:.3f}]  "
          f"omega2: [{omega2[0]:.3f}, {omega2[-1]:.3f}]")

    out = out_dir or str(ROOT / cfg.get("output", "plot_dir", fallback="plots"))
    tag = Path(ckpt_path).stem

    plot_comparisons(G_batch, A_true_batch, A_pred_batch,
                     dw1, dw2, out, tag, omega1, omega2, params_list)
    print("Evaluation done.")


# ─────────────────────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ACANN_v3 评估脚本")
    parser.add_argument(
        "--ckpt",
        default=str(ROOT / "checkpoints_2.2" / "best.pth"),
        help="checkpoint 路径",
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "config" / "train_config.ini"),
        help="训练配置文件路径",
    )
    parser.add_argument(
        "--n", type=int, default=8,
        help="评估样本数",
    )
    parser.add_argument(
        "--out", default=None,
        help="图片输出目录",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机种子",
    )
    args = parser.parse_args()
    evaluate(args.ckpt, args.config, args.n, args.out, args.seed)
