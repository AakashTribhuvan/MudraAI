"""Load a scikit-learn ``.joblib`` model without trusting it.

joblib files are pickles: loading one normally runs whatever code it names. This loader resolves only an
allow-list of numpy / scikit-learn / joblib-array classes and refuses everything else, so a tampered file
fails to load instead of running code ("Restricting Globals", Python pickle docs). Used for the teammate's
SVM models (AakashTribhuvan/MitIndia, branch work/static-landmark-tracking, commit 8105c5e).
"""

from __future__ import annotations

import pickle
from pathlib import Path

from joblib import numpy_pickle

ALLOWED_MODULE_PREFIXES = ("numpy.", "sklearn.", "scipy.sparse.")
ALLOWED = {
    ("numpy", "dtype"), ("numpy", "ndarray"),
    ("joblib.numpy_pickle", "NumpyArrayWrapper"),
    ("collections", "OrderedDict"),
}


class _RestrictedUnpickler(numpy_pickle.NumpyUnpickler):
    def find_class(self, module: str, name: str):
        if (module, name) in ALLOWED or module.startswith(ALLOWED_MODULE_PREFIXES):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"refused to load {module}.{name} from a model file")


def load(path: Path):
    """Unpickle a joblib file (uncompressed) allowing only numpy / scikit-learn objects."""
    path = Path(path)
    with path.open("rb") as stream:
        try:
            unpickler = _RestrictedUnpickler(str(path), stream, ensure_native_byte_order=True, mmap_mode=None)
        except TypeError:  # joblib < 1.5 has no ensure_native_byte_order
            unpickler = _RestrictedUnpickler(str(path), stream, mmap_mode=None)
        return unpickler.load()
