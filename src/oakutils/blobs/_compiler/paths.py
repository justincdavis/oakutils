import os
import sysconfig
import shutil


def get_site_packages_path() -> str:
    """
    Gets the path to the site-packages folder.

    Returns
    -------
    str
        The path to the site-packages folder.
    """
    return os.path.abspath(sysconfig.get_paths()["purelib"])


def get_oakutils_path() -> str:
    """
    Gets the path to the oakutils folder.

    Returns
    -------
    str
        The path to the oakutils folder.
    """
    return os.path.join(get_site_packages_path(), "oakutils")


def get_blobs_path() -> str:
    """
    Gets the path to the oakutils blobs folder.

    Returns
    -------
    str
        The path to the oakutils blobs folder.
    """
    return os.path.join(get_oakutils_path(), "blobs")


def get_cache_dir_path() -> str:
    """
    Gets the path to the oakutils blobs cache folder.

    Returns
    -------
    str
        The path to the oakutils blobs cache folder.
    """
    return os.path.join(get_blobs_path(), "_cache")


def delete_folder(folder_path: str):
    """
    Deletes the folder at the given path.
    This will delete all files and subfolders in the folder,
    as well as the folder itself.
    This is permanent and cannot be undone.

    Parameters
    ----------
    folder_path : str
        The path to the folder to delete.
    """
    if not os.path.exists(folder_path):
        return
    for root, dirs, files in os.walk(folder_path):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))
    os.rmdir(folder_path)
