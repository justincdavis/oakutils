# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
from __future__ import annotations

import os
import site
import sysconfig
from pathlib import Path


def get_site_packages_path() -> Path:
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
    return Path(site_packages).resolve()


def get_oakutils_path() -> Path:
    """
    Use to get the path to the oakutils folder.

    Returns
    -------
    str
        The path to the oakutils folder.
    """
    return Path(get_site_packages_path()) / "oakutils"


def get_blobs_path() -> Path:
    """
    Use to get the path to the oakutils blobs folder.

    Returns
    -------
    str
        The path to the oakutils blobs folder.
    """
    return Path(get_oakutils_path()) / "blobs"


def get_cache_dir_path() -> Path:
    """
    Use to get the path to the oakutils blobs cache folder.

    Returns
    -------
    str
        The path to the oakutils blobs cache folder.
    """
    return Path(get_blobs_path()) / "_cache"


def get_models_dir_path() -> Path:
    """
    Use to get the path to the oakutils blobs models folder.

    Returns
    -------
    str
        The path to the oakutils blobs models folder.
    """
    return Path(get_blobs_path()) / "models"


def delete_folder(directory: Path) -> None:
    """
    Use to delete a folder and all of its contents.

    Parameters
    ----------
    directory : Path
        The path to the folder to delete.
    """
    for item in directory.iterdir():
        if item.is_dir():
            delete_folder(item)
        else:
            item.unlink()
    directory.rmdir()


def clear_cache() -> None:
    """Use to clear the cache folder for the oakutils blobs."""
    cache_dir = get_cache_dir_path()
    delete_folder(cache_dir)
    cache_dir.mkdir()
