"""
PyTorch Dataset：从 memmap 加载谱函数 A，从 CSV 加载输入 G（Chebyshev 系数）。

支持 target_size 参数：若训练分辨率低于数据分辨率，可在 __getitem__ 中双线性缩放 A，
无需重新生成数据。
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Tuple


class Database(Dataset):
    """
    Parameters
    ----------
    memmap_meta_path : str | Path
        由 save_A_memmap 生成的 *_meta.npz 路径。
    csv_input : str | Path
        Chebyshev 系数文件（nl_*.csv）。
    nb_data : int
        最多使用的样本数。
    target_size : (int, int) | None
        训练时的目标分辨率 (w1, w2)。若 None 则使用 memmap 原始分辨率。
        缩放在 CPU 上以双线性插值完成，不影响磁盘数据。
    transform : callable | None
        额外数据增强，签名 (g, a) -> (g, a)。
    """

    def __init__(
        self,
        memmap_meta_path: str,
        csv_input: str,
        nb_data: int = 2500,
        target_size: Optional[Tuple[int, int]] = None,
        transform=None,
    ):
        self.transform   = transform
        self.target_size = target_size

        # ── 读 memmap meta ────────────────────────────────────────────────
        meta    = np.load(memmap_meta_path, allow_pickle=True)
        N_total, N1, N2 = map(int, meta["shape"])
        cursor  = int(meta["cursor"][0])
        dtype   = np.dtype(str(meta["dtype"]))

        self.N_use = min(nb_data, cursor)
        if self.N_use < nb_data:
            print(f"[Dataset] nb_data={nb_data} 但 memmap 只有 {cursor} 条，"
                  f"实际用 {self.N_use} 条")

        # ── 推断 .dat 路径 ────────────────────────────────────────────────
        meta_path = Path(memmap_meta_path)
        dat_path  = meta_path.with_name(
            meta_path.name.replace("_meta.npz", ".dat")
        )

        # ── A：只 memmap，不全量 copy ─────────────────────────────────────
        self.A_mm    = np.memmap(dat_path, dtype=dtype, mode="r",
                                 shape=(N_total, N1, N2))
        self.A_shape = (N1, N2)

        # ── G：全量读入内存（float32）────────────────────────────────────
        raw = np.genfromtxt(csv_input, delimiter=",", max_rows=self.N_use)
        if raw.ndim == 1:
            raw = raw[None, :]
        self.G = torch.tensor(raw[:self.N_use], dtype=torch.float32)

        if self.target_size is not None:
            print(f"[Dataset] A 将在加载时缩放 {(N1, N2)} → {self.target_size}")

    def __len__(self) -> int:
        return self.N_use

    def __getitem__(self, idx: int):
        g = self.G[idx]                                         # (nb_gc,)

        a = torch.from_numpy(
            np.array(self.A_mm[idx], copy=False).astype(np.float32)
        )                                                       # (N1, N2)

        # 按需缩放到训练分辨率
        # 双线性插值保值不保积分点数，但物理积分 ∫∫A dω₁dω₂ 天然守恒
        # （dw 随分辨率变大刚好补偿 sum 变小），无需手动 renorm
        if self.target_size is not None and self.target_size != self.A_shape:
            a = F.interpolate(
                a[None, None],                                  # (1,1,N1,N2)
                size=self.target_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze()                                         # (w1, w2)
            a = F.relu(a)

        if self.transform:
            g, a = self.transform((g, a))

        return g, a
