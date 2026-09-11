#!/usr/bin/env python3
"""Calculate energies for VASP POSCAR images with MACE."""

import argparse
import csv
from pathlib import Path

from mace.calculators import MACECalculator
from ase.constraints import FixAtoms
from ase.io import read


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--pattern", default="POSCAR_*")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="neb_energies.csv", type=Path)
    args = parser.parse_args()

    images_dir = args.images_dir.resolve()
    model = args.model.resolve()
    images = sorted(p for p in images_dir.glob(args.pattern) if p.is_file())
    if not images:
        raise SystemExit(f"No images matched {args.pattern!r} in {images_dir}")
    if not model.is_file():
        raise SystemExit(f"Model not found: {model}")

    calc = MACECalculator(
        model_paths=str(model), device=args.device, default_dtype="float32"
    )
    rows = []
    for image_path in images:
        atoms = read(str(image_path), format="vasp")
        atoms.set_constraint(FixAtoms(indices=list(range(len(atoms)))))
        atoms.calc = calc
        energy = float(atoms.get_potential_energy())
        rows.append((image_path.name, energy))
        print(f"{image_path.name}: {energy:.6f} eV")

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "energy_eV"])
        writer.writerows(rows)
    print(f"Wrote {len(rows)} energies to {output}")


if __name__ == "__main__":
    main()
