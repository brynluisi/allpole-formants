"""Fetch the released LP-LSTM / SMELP checkpoints from the Hugging Face Hub.

Weights live at https://huggingface.co/Aalto-Speech-Synthesis/smelp . Only the well-known
default path checkpoints/<model>.pt is ever auto-downloaded; a checkpoint path you give
explicitly is never silently substituted -- if it's missing, that's still an error.
"""
import os
import shutil
import ssl
import urllib.request

try:
    import certifi
    _CONTEXT = ssl.create_default_context(cafile=certifi.where())
except ImportError:  # falls back to the system CA store (may fail on macOS's python.org builds)
    _CONTEXT = None

HF_REPO = "Aalto-Speech-Synthesis/smelp"
FILENAMES = {"lp_lstm": "lp_lstm.pt", "smelp": "smelp.pt"}
DEFAULT_PATHS = {model: f"checkpoints/{filename}" for model, filename in FILENAMES.items()}


def hf_url(model: str) -> str:
    return f"https://huggingface.co/{HF_REPO}/resolve/main/{FILENAMES[model]}"


def _download(url: str, path: str) -> None:
    """Stream `url` to a temp file next to `path`, then rename -- no partial file on failure."""
    tmp_path = path + ".part"
    try:
        with urllib.request.urlopen(url, context=_CONTEXT) as response, open(tmp_path, "wb") as f:
            shutil.copyfileobj(response, f)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def ensure_checkpoint(model: str, path: str, flag: str, raise_on_error: bool = True) -> str:
    """Download `model`'s checkpoint straight to `path`, but only if `path` is the default
    checkpoints/<model>.pt location and nothing is there yet. Any other missing path is left
    for the caller to report -- it's a user-given path, not ours to fill in.

    On a download failure, `raise_on_error` (the default) stops right there -- right for a
    single-checkpoint caller. A caller checking several models at once (so it can report every
    problem together, not just the first) passes `raise_on_error=False`: the failure is printed,
    `path` is returned still missing, and it's up to that caller to notice and report it.
    """
    if os.path.isfile(path) or path != DEFAULT_PATHS[model]:
        return path
    url = hf_url(model)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    print(f"Downloading {model} checkpoint from {url} ...", flush=True)
    try:
        _download(url, path)
    except Exception as err:
        message = (f"Could not download {url}: {err}\n"
                   f"Download it yourself and pass it with {flag}, or place it at {path}.")
        if raise_on_error:
            raise SystemExit(message)
        print(message, flush=True)
    return path
