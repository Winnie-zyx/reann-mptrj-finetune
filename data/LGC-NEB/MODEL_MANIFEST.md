# 保留的势函数清单

| 清理后目录 | 原始来源代表 | 文件 |
|---|---|---|
| `potentials/reann-mptrj-sse-ft2/` | `path1-1/MPtrj-SSE/finetune-2/` | `REANN_PES_DOUBLE.pt`, `REANN_PES_FLOAT.pt`, `REANN.pth` |
| `potentials/reann-mptrj-lgc-ft2/` | `path1-1/MPtrj-LGC/finetune-2/` | `REANN_PES_DOUBLE.pt`, `REANN_PES_FLOAT.pt`, `REANN.pth` |
| `potentials/reann-mptrj-wbm-ft2/` | `path1-1/MPtrj-WBM/finetune-2/` | `REANN_PES_DOUBLE.pt`, `REANN_PES_FLOAT.pt`, `REANN.pth` |
| `potentials/reann-mptrj-lgc-mlmd-sample-ft2/` | `path1-1/MPtrj-LGC-MLMD-sample/finetune-2/` | `REANN_PES_DOUBLE.pt`, `REANN_PES_FLOAT.pt`, `REANN.pth` |
| `potentials/reann-mptrj-lgc-1ps-250c-ft2/` | `path1-1/MPtrj-LGC-1ps-250c/finetune-2/` | `REANN_PES_DOUBLE.pt`, `REANN_PES_FLOAT.pt`, `REANN.pth` |
| `potentials/mace-mp-0/` | `path1-1/MACE-MP-0/` | `2023-12-03-mace-128-L1_epoch-199.model` |
| `potentials/nequip-mptrj/` | `path3-1/nequip-MPtrj/` | `deployed.pth` |

同一模型在不同 `path*` 目录中的副本经过内容校验后只保留一份。`finetune-2-1ps-250c` 与 `finetune-2` 的模型内容一致，因此未重复保留。
