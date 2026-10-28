"""Fashion-MNIST loading with checksum verification.

Reads the official IDX files (gzip) from ``data/fashion`` and downloads them
from the official zalandoresearch/fashion-mnist GitHub repository if missing.
Every file is checked against the MD5 checksum published in that repository's
README, so results are reproducible and a corrupted download is caught early.
"""

from __future__ import annotations

import gzip
import hashlib
import urllib.request
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "fashion"

BASE_URL = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/"

# MD5 checksums published in https://github.com/zalandoresearch/fashion-mnist (README)
FILES = {
    "train-images-idx3-ubyte.gz": "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
    "train-labels-idx1-ubyte.gz": "25c81989df183df01b3e8a0aad5dffbe",
    "t10k-images-idx3-ubyte.gz": "bef4ecab320f06d8554ea6380940ec79",
    "t10k-labels-idx1-ubyte.gz": "bb300cfdad3c16e7a12a480ee83cd310",
}

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_downloaded(data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    """Make sure all four files exist and match their published MD5."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, expected in FILES.items():
        path = data_dir / name
        if not path.exists():
            print(f"Downloading {name} ...")
            urllib.request.urlretrieve(BASE_URL + name, path)
        actual = md5sum(path)
        if actual != expected:
            raise RuntimeError(
                f"MD5 mismatch for {path}: expected {expected}, got {actual}. "
                "Delete the file and run again."
            )
    return data_dir


def _read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        data = f.read()
    magic = int.from_bytes(data[0:4], "big")
    ndim = magic & 0xFF
    dims = [int.from_bytes(data[4 + 4 * i : 8 + 4 * i], "big") for i in range(ndim)]
    offset = 4 + 4 * ndim
    return np.frombuffer(data, dtype=np.uint8, offset=offset).reshape(dims)


def load_split(split: str, data_dir: Path = DEFAULT_DATA_DIR):
    """Return (images uint8 [N,1,28,28], labels int64 [N]) for 'train' or 'test'."""
    data_dir = ensure_downloaded(data_dir)
    prefix = {"train": "train", "test": "t10k"}[split]
    images = _read_idx(data_dir / f"{prefix}-images-idx3-ubyte.gz")
    labels = _read_idx(data_dir / f"{prefix}-labels-idx1-ubyte.gz")
    return images[:, None, :, :].copy(), labels.astype(np.int64)


def to_float(images_uint8: np.ndarray) -> np.ndarray:
    """uint8 [0,255] -> float32 [0,1], same as torchvision ToTensor (and FINN's ToTensor pre-proc)."""
    return images_uint8.astype(np.float32) / 255.0
