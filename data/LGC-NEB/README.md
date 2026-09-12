# LGC-NEB

```text
data/LGC-NEB/
├── neb_images/
│   ├── path1-1/
│   │   └── POSCAR_00 ... POSCAR_06
│   ├── path1-2/
│   │   └── POSCAR_00 ... POSCAR_06
│   ├── path2-1/
│   │   └── POSCAR_00 ... POSCAR_04
│   ├── path2-2/
│   │   └── POSCAR_00 ... POSCAR_04
│   ├── path3-1/
│   │   └── POSCAR_00 ... POSCAR_04
│   ├── path3-2/
│   │   └── POSCAR_00 ... POSCAR_04
│   └── path3-3/
│       └── POSCAR_00 ... POSCAR_04
├── potentials/
│   ├── reann-mptrj-sse-ft2/
│   │   ├── REANN.pth
│   │   ├── REANN_PES_DOUBLE.pt
│   │   └── REANN_PES_FLOAT.pt
│   ├── reann-mptrj-lgc-ft2/
│   │   ├── REANN.pth
│   │   ├── REANN_PES_DOUBLE.pt
│   │   └── REANN_PES_FLOAT.pt
│   ├── reann-mptrj-wbm-ft2/
│   │   ├── REANN.pth
│   │   ├── REANN_PES_DOUBLE.pt
│   │   └── REANN_PES_FLOAT.pt
│   ├── reann-mptrj-lgc-mlmd-sample-ft2/
│   │   ├── REANN.pth
│   │   ├── REANN_PES_DOUBLE.pt
│   │   └── REANN_PES_FLOAT.pt
│   ├── reann-mptrj-lgc-1ps-250c-ft2/
│   │   ├── REANN.pth
│   │   ├── REANN_PES_DOUBLE.pt
│   │   └── REANN_PES_FLOAT.pt
│   ├── mace-mp-0/
│   │   └── 2023-12-03-mace-128-L1_epoch-199.model
│   └── nequip-mptrj/
│       └── deployed.pth
├── scripts/
│   ├── predict_neb_reann.py
│   ├── predict_neb_mace.py
│   ├── predict_neb_nequip.py
│   └── original/
│       ├── static_all_reann.py
│       ├── static_mace.py
│       └── static_nequip.py
├── finetuning/
│   ├── configurations/
│   │   ├── configuration-LGC-AIMDsample-250
│   │   └── configuration-MLMD-sample
│   └── model_parameters/
│       └── PES-LGC-AIMD-FT2.zip
├── analysis/
│   ├── scripts/
│   │   └── similarity-tsne.py
│   └── data/
│       └── 11和19/
│           └── 工作簿1.xlsx
├── MODEL_MANIFEST.md
├── README.md
└── requirements.txt
```
