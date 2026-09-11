# Training configuration

`src/read.py` expects the following files relative to the repository root:

- `para/input_nn`
- `para/input_density`

The supplied archive did not include these files. Add the project-specific model, species, cutoff, optimizer, and data settings here before launching `run/train.py`. Keep private dataset paths and credentials out of this directory.
