"""Shared test helpers -- NOT a test module itself (no Test* classes).

Provides a mock `folder_paths` module (ComfyUI's real module isn't
importable outside a running ComfyUI instance) and a small loader for
importing a node .py file by its actual on-disk path, since every node
lives in a directory whose name contains spaces (e.g. "08 Model
Validation") and can't be imported as a normal dotted package.
"""
import importlib.util
import os
import sys
import tempfile

_REPO_PY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "py")

_MOCK_ENV_CANDIDATES = [
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                 "pilot_scripts", "mock_comfy_env"),
]
for _p in _MOCK_ENV_CANDIDATES:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import folder_paths  # noqa: E402

_OUTPUT_ROOT = None


def fresh_output_dir():
    """A brand-new temp directory, set as folder_paths' output root, so
    each test gets an isolated output tree (no cross-test file leakage)."""
    global _OUTPUT_ROOT
    _OUTPUT_ROOT = tempfile.mkdtemp(prefix="comfyqsar_test_")
    folder_paths.set_output_directory(_OUTPUT_ROOT)
    return _OUTPUT_ROOT


def load_node_module(track, node_dir, filename, alias=None):
    """Import <repo>/py/<track>/<node_dir>/<filename> as a fresh module,
    isolated from any other node file of the same name already imported
    (several node directories legitimately use the same filename, e.g.
    every "Model_Validation.py"). `alias` only picks the synthetic module
    name passed to spec_from_file_location (used in error messages/repr);
    it is NOT registered in sys.modules.

    Deliberately mirrors the real ComfyUI loader (code/__init__.py's
    _load()), which also never does sys.modules[module_name] = mod --
    registering it here would be MORE hospitable to the module than
    production actually is, which once hid a real-vs-test discrepancy: a
    joblib/loky worker pickling a function from one of these node files
    behaves differently depending on whether sys.modules has a matching
    entry (cloudpickle pickles by reference, which then fails to
    re-import in a fresh worker process, vs. by value, which works) --
    see progress notes on the num_cores/loky investigation."""
    path = os.path.join(_REPO_PY, track, node_dir, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    module_name = alias or f"{track}_{node_dir}_{filename}".replace(" ", "_").replace(".", "_")
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_module_by_relpath(rel_path, alias=None):
    """Import <repo>/py/<rel_path> as a fresh module -- for files that
    don't fit the <track>/<node_dir>/<filename> shape (e.g. Screener/
    custom_user_screener.py, which has no per-node subdirectory).

    Does not register in sys.modules -- see load_node_module's docstring
    for why this matters (matches the real ComfyUI loader exactly)."""
    path = os.path.join(_REPO_PY, rel_path)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    module_name = alias or rel_path.replace(os.sep, "_").replace(" ", "_").replace(".", "_")
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rdkit_available():
    try:
        import rdkit  # noqa: F401
        return True
    except ImportError:
        return False


def padelpy_available():
    try:
        import padelpy  # noqa: F401
        return True
    except ImportError:
        return False
