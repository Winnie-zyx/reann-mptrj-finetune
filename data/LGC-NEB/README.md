# LGC NEB — FT-2 精简包

这是从 `LGC-NEB (2).zip` 整理出的 NEB 数据精简包，重点保留各类 `finetune-2`（FT-2）势函数及 7 条 NEB 路径的结构文件。

## 保留内容

- `neb_images/`：7 条路径的 NEB 图像，保留原始 `POSCAR_*` 文件
- `potentials/`：每类势函数只保留一份
  - 5 套 FT-2 REANN 模型：MPtrj-SSE、MPtrj-LGC、MPtrj-WBM、MPtrj-LGC-MLMD-sample、MPtrj-LGC-1ps-250c
  - 1 套 MACE-MP-0 模型
  - 1 套 NequIP-MPtrj 模型
- `scripts/`：用于批量读取 NEB 图像并调用势函数计算能量的程序
- `scripts/original/`：原压缩包中保留的代表性预测脚本

`MPtrj-LGC-1ps-250c/finetune-2-1ps-250c` 与 `finetune-2` 的模型内容重复，已合并为一份。不同 `path*` 文件夹中的同一模型也只保留一份。

## 计算 NEB 图像能量

推荐使用新的通用脚本，从本目录运行：

```bash
python scripts/predict_neb_reann.py \
  --images-dir neb_images/path1-1 \
  --model potentials/reann-mptrj-lgc-ft2/REANN_PES_DOUBLE.pt
```

MACE 和 NequIP 分别使用：

```bash
python scripts/predict_neb_mace.py \
  --images-dir neb_images/path1-1 \
  --model potentials/mace-mp-0/2023-12-03-mace-128-L1_epoch-199.model

python scripts/predict_neb_nequip.py \
  --images-dir neb_images/path3-1 \
  --model potentials/nequip-mptrj/deployed.pth
```

脚本默认生成 `neb_energies.csv`，也可以通过 `--output` 指定结果文件。REANN 脚本沿用原工程的 `ase.calculators.reann_z.REANN` 计算器；MACE 和 NequIP 需要各自的 ASE 计算器依赖。

## 清理说明

已移除各路径重复的模型、副本目录、scratch/非 FT-2 版本、MACE 运行日志/轨迹和编辑器临时文件。原始 ZIP 文件未修改。
