# ACANN

复频域谱函数重建的 ACANN-CNN 模型及相关工具。

## 项目结构

```
ACANN/
├── config/           # 配置文件
│   └── params.ini    # 数据生成与训练参数
├── models/           # 模型定义
│   └── acann_cnn.py  # ACANN-CNN 网络
├── data/             # 数据处理
│   ├── dataset.py    # PyTorch Dataset（memmap + CSV）
│   └── save_memmap.py # 谱函数 memmap 保存工具
├── scripts/          # 可执行脚本
│   ├── generate_data_jax.py     # JAX 数据生成
│   ├── generate_data_pytorch.py # PyTorch 数据生成
│   └── train.py                 # 训练脚本
├── utils/            # 工具函数
│   └── rolling_mean.py
└── README.md
```

## 使用说明

1. **配置**：编辑 `config/params.ini`，设置 `tau_grid` 为虚时网格文件路径（如 `tgrid.dat`）。

2. **生成数据**：
   ```bash
   python scripts/generate_data_pytorch.py   # 或 generate_data_jax.py
   ```
   输出默认在 `data_output/` 目录。

3. **训练**：
   ```bash
   python scripts/train.py
   ```
   需在 `train.py` 中配置训练数据路径（`traindata/` 等）及 checkpoint 目录。
