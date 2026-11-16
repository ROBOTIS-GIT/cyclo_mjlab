# Copyright 2026 ROBOTIS CO., LTD.
#
# Author: Insu Park

"""Check the installed runtime against this checkout's requirements."""

import argparse
import importlib
import importlib.metadata as metadata
from pathlib import Path
import sys


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--gpu", action="store_true", help="Also check CUDA execution.")
  args = parser.parse_args()

  try:
    # Both the previous Python 3.11 image and Python 3.12 provide tomllib.
    import tomllib
    from packaging.requirements import Requirement
    from packaging.specifiers import SpecifierSet

    root = Path(__file__).resolve().parents[1]
    with (root / "pyproject.toml").open("rb") as stream:
      project = tomllib.load(stream)["project"]

    print(f"[INFO] Checkout release: {project['version']}")
    python_version = ".".join(map(str, sys.version_info[:3]))
    errors = []
    if python_version not in SpecifierSet(project["requires-python"]):
      errors.append(f"Python {python_version}; required {project['requires-python']}")
    else:
      print(f"[OK] Python {python_version}")

    for entry in project["dependencies"]:
      requirement = Requirement(entry)
      if requirement.marker and not requirement.marker.evaluate():
        continue
      try:
        actual = metadata.version(requirement.name)
      except metadata.PackageNotFoundError:
        errors.append(f"{requirement.name} is not installed; required {requirement}")
        continue
      if actual not in requirement.specifier:
        errors.append(f"{requirement.name} {actual}; required {requirement.specifier}")
      else:
        print(f"[OK] {requirement.name} {actual}")

    if errors:
      raise RuntimeError("\n".join(errors))

    # Version metadata alone does not prove that binary dependencies can load.
    for name in ("mjlab", "mujoco", "mujoco_warp", "warp", "torch"):
      importlib.import_module(name)

    if args.gpu:
      import torch

      if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; check the NVIDIA driver and Container Toolkit.")
      result = (torch.ones(4, device="cuda:0") * 2).sum().item()
      if result != 8:
        raise RuntimeError("CUDA tensor computation failed.")
      print(f"[OK] CUDA execution: {torch.cuda.get_device_name(0)}")
  except Exception as exc:
    print(f"[ERROR] Environment check failed: {exc}", file=sys.stderr)
    return 1

  print("[OK] Environment check passed.")
  return 0


if __name__ == "__main__":
  sys.exit(main())
