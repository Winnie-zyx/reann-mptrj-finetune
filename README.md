# REANN-MPtrj fine-tuning model

This repository contains the REANN model source, inference interfaces, LAMMPS/ASE examples, a TorchScript model checkpoint, and the bundled manual from the supplied archive.

## Contents

- `src/`: model, data loading, loss, and optimization code
- `run/`: training entry points
- `ASE/`: ASE calculator and test examples
- `lammps-interface/` and `lammps-REANN-interface/`: LAMMPS integration examples
- `inference/`: inference utilities
- `ASE/calculators/test/REANN_PES_DOUBLE.pt`: TorchScript checkpoint used by the ASE interface
- `manual/manual.pdf`: original manual

Obsolete dated/backup variants and the broken legacy `run/test_train.py` artifact were left out of the published tree. The original ZIP archive remains untouched on the desktop.

## Fine-tuning

The training entry point is `run/train.py`. It reads configuration from:

- `para/input_nn`
- `para/input_density`

and expects training and validation data under `train/` and `test/` in the format handled by `src/read_data.py`. These files and datasets were not present in the supplied archive, so they must be added for a complete training run. See `para/README.md` and `manual/manual.pdf` before preparing them.

After installing the Python dependencies, run from the repository root, for example:

```bash
python -m pip install -r requirements.txt
torchrun --standalone --nproc_per_node=1 run/train.py
```

For multi-GPU training, increase `--nproc_per_node` and use a suitable distributed backend in `para/input_nn`.

## ASE inference

`reann.py` loads a TorchScript checkpoint and exposes a calculator for energy and forces. Update the atom types, input structure, and checkpoint path (the bundled checkpoint is under `ASE/calculators/test/`) in the example before running it. The bundled examples contain machine-specific paths and should be adapted to the local environment.

## LAMMPS interface

The LAMMPS interface requires a compatible LAMMPS installation and a C++/CMake toolchain. The checked-in examples are source examples only; local build outputs were intentionally excluded.

## Provenance

Source: `reann_test2502_stress_add.zip`, supplied by the repository owner. No license file was included in the archive; add the appropriate license and citation information before making the repository public.
