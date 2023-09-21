from __future__ import annotations

import os
import site
import sysconfig


def get_site_packages_path() -> str:
    """
    Use to get the path to the site-packages folder.

    Returns
    -------
    str
        The path to the site-packages folder.
    """
    site_site_packages = site.getusersitepackages()
    sysconfig_site_packages = sysconfig.get_paths()["purelib"]
    site_packages = (
        site_site_packages if os.name == "posix" else sysconfig_site_packages
    )
    return os.path.abspath(site_packages)


def get_oakutils_path() -> str:
    """
    Use to get the path to the oakutils folder.

    Returns
    -------
    str
        The path to the oakutils folder.
    """
    return os.path.join(get_site_packages_path(), "oakutils")


def get_blobs_path() -> str:
    """
    Use to get the path to the oakutils blobs folder.

    Returns
    -------
    str
        The path to the oakutils blobs folder.
    """
    return os.path.join(get_oakutils_path(), "blobs")


def get_cache_dir_path() -> str:
    """
    Use to get the path to the oakutils blobs cache folder.

    Returns
    -------
    str
        The path to the oakutils blobs cache folder.
    """
    return os.path.join(get_blobs_path(), "_cache")


def get_models_dir_path() -> str:
    """
    Use to get the path to the oakutils blobs models folder.

    Returns
    -------
    str
        The path to the oakutils blobs models folder.
    """
    return os.path.join(get_blobs_path(), "models")
