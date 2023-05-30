from typing import Dict


def dict_to_str(d: Dict) -> str:
    """
    Converts a dictionary to a string by combining the values with underscores.

    Parameters
    ----------
    d : Dict
        The dictionary to convert

    Returns
    -------
    str
        The converted string
    """
    rv = "".join([f"{str(v)}x{str(v)}_" if "kernel_size" in k else f"{str(v)}_" for k, v in d.items()])
    rv = rv.removesuffix("_")
    return rv
