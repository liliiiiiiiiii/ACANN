"""
将谱函数 A 保存为 memmap 格式，支持增量写入。
"""
from pathlib import Path
import numpy as np


def save_A_memmap(
    A_batch,
    omega1_grid=None,
    omega2_grid=None,
    base_dir="../Database",
    name="A_train",
    N_total=10000,          # 最终总样本数（建议你确定一个）
    dtype=np.float32
):
    """
    A_batch: (B, 250, 250)  你本次生成的一批数据
    omega1_grid/omega2_grid: (250,) 可选；第一次创建时建议传入，用于保存网格坐标
    多次运行会自动从上次 cursor 继续写入
    """
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)

    data_path = base / f"{name}.dat"
    meta_path = base / f"{name}_meta.npz"

    A_batch = np.asarray(A_batch, dtype=dtype)
    if A_batch.ndim != 3:
        raise ValueError(f"A_batch must be 3D (B,N1,N2), got {A_batch.shape}")

    B, N1, N2 = A_batch.shape

    # ---- 若 meta 不存在：自动初始化 ----
    if not meta_path.exists() or not data_path.exists():
        # 预分配 (N_total, N1, N2)
        mm = np.memmap(data_path, dtype=dtype, mode="w+", shape=(N_total, N1, N2))
        mm.flush()

        np.savez(
            meta_path,
            shape=np.array([N_total, N1, N2], dtype=np.int32),
            dtype=str(np.dtype(dtype)),
            cursor=np.array([0], dtype=np.int32),
            omega1=np.array(omega1_grid, dtype=np.float32) if omega1_grid is not None else None,
            omega2=np.array(omega2_grid, dtype=np.float32) if omega2_grid is not None else None,
        )

    # ---- 读取 meta，继续写 ----
    meta = np.load(meta_path, allow_pickle=True)
    N_total_m, N1_m, N2_m = map(int, meta["shape"])
    cursor = int(meta["cursor"][0])
    dtype_m = np.dtype(str(meta["dtype"]))

    if (N1_m, N2_m) != (N1, N2):
        raise ValueError(f"Shape mismatch: storage expects {(N1_m,N2_m)}, but got {(N1,N2)}")
    if dtype_m != np.dtype(dtype):
        # 允许你传 dtype 不一致时也能强制转换
        A_batch = A_batch.astype(dtype_m, copy=False)

    # 如果超出容量：只写得下的部分（避免崩）
    remaining = N_total_m - cursor
    if remaining <= 0:
        print(f"[{name}] FULL: already have {cursor}/{N_total_m}. Nothing written.")
        return cursor, cursor

    write_B = min(B, remaining)

    mm = np.memmap(data_path, dtype=dtype_m, mode="r+", shape=(N_total_m, N1_m, N2_m))
    mm[cursor:cursor + write_B] = A_batch[:write_B]
    mm.flush()

    new_cursor = cursor + write_B

    # 更新 meta（npz 不能原地改，直接重写）
    omega1 = meta["omega1"] if "omega1" in meta.files else None
    omega2 = meta["omega2"] if "omega2" in meta.files else None
    np.savez(
        meta_path,
        shape=meta["shape"],
        dtype=str(dtype_m),
        cursor=np.array([new_cursor], dtype=np.int32),
        omega1=omega1,
        omega2=omega2,
    )

    print(f"[{name}] wrote {write_B} samples, cursor: {cursor} -> {new_cursor} (total {N_total_m})")
    return cursor, new_cursor
