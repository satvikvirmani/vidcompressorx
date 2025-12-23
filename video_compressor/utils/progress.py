from typing import Iterable, Optional

def _in_notebook() -> bool:
    try:
        from IPython import get_ipython
        ip = get_ipython()
        return ip is not None and "IPKernelApp" in ip.config
    except Exception:
        return False


def get_tqdm():
    if _in_notebook():
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
    return tqdm


def progress(
    iterable: Iterable,
    desc: Optional[str] = None,
    total: Optional[int] = None,
):
    tqdm = get_tqdm()
    return tqdm(iterable, desc=desc, total=total)