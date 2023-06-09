from typing import Dict


# for 3.8 compatibility
def remove_suffix(input_string: str, suffix: str) -> str:
    if suffix and input_string.endswith(suffix):
        return input_string[:-len(suffix)]
    return input_string


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
    rv = "".join(
        [
            f"{str(v)}x{str(v)}_" if "kernel_size" in k else f"{str(v)}_"
            for k, v in d.items()
        ]
    )
    rv = remove_suffix(rv, "_")
    return rv
