"""Check whether the local environment can run the project."""

from __future__ import annotations

import importlib.util
import sys


REQUIRED_PACKAGES = [
    "torch",
    "torchaudio",
    "librosa",
    "soundfile",
    "numpy",
    "scipy",
    "tqdm",
    "matplotlib",
    "PyQt5",
]

OPTIONAL_PACKAGES = [
    "pystoi",
    "pesq",
]


def package_available(package_name: str) -> bool:
    """Return True when a package can be imported."""

    return importlib.util.find_spec(package_name) is not None


def main() -> int:
    """Print environment status and return a process exit code."""

    print(f"Python: {sys.version.split()[0]}")
    missing_required: list[str] = []
    for package_name in REQUIRED_PACKAGES:
        status = "ok" if package_available(package_name) else "missing"
        print(f"{package_name}: {status}")
        if status == "missing":
            missing_required.append(package_name)

    for package_name in OPTIONAL_PACKAGES:
        status = "ok" if package_available(package_name) else "missing"
        print(f"{package_name}: {status} (optional)")

    try:
        import torch

        print(f"torch cuda available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"torch cuda device: {torch.cuda.get_device_name(0)}")
    except Exception as exc:
        print(f"torch cuda check failed: {exc}")

    if missing_required:
        print()
        print("Missing required packages. Run scripts\\setup_windows.bat first.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

