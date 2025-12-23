from typing import Any

def _in_notebook() -> bool:
    try:
        from IPython import get_ipython
        ip = get_ipython()
        return ip is not None and "IPKernelApp" in ip.config
    except Exception:
        return False


def get_tqdm():
    """
    Returns tqdm or tqdm.notebook.tqdm depending on environment.
    """
    if _in_notebook():
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
    return tqdm


def progress(*args: Any, **kwargs: Any):
    """
    Drop-in replacement for tqdm that works in scripts and notebooks.
    """
    tqdm = get_tqdm()
    return tqdm(*args, **kwargs)