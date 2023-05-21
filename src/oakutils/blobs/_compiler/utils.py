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
    return "".join([str(v) + "_" for v in d.values()]).removesuffix("_")
