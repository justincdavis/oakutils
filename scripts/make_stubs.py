import subprocess
import os
import shutil

import open3d


def get_name(module):
    try:
        return module.__name__
    except AttributeError:
        return module

def move_open3d_stubs():
    # moves stubs from out to typings
    delete_folder(os.path.join(os.path.dirname(__file__), "..", "typings", "open3d"))
    shutil.move(
        os.path.join(os.path.dirname(__file__), "..", "out", "open3d"),
        os.path.join(os.path.dirname(__file__), "..", "typings", "open3d"),
    )

def delete_folder(folder_path: str):
    """
    Deletes the folder at the given path.
    """
    if not os.path.exists(folder_path):
        return
    for root, dirs, files in os.walk(folder_path):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))
    os.rmdir(folder_path)


def fix_open3d_stubs():
    folder = os.path.join(os.path.dirname(__file__), "..", "out", "open3d")
    pybind_folder = os.path.join(folder, "cpu", "pybind")
    for file in os.listdir(pybind_folder):
        folder_name = file.split(".")[0]
        delete_folder(os.path.join(folder, folder_name))
        os.makedirs(os.path.join(folder, folder_name), exist_ok=True)
        shutil.move(
            os.path.join(pybind_folder, file),
            os.path.join(folder, folder_name, "__init__.pyi"),
        )
    delete_folder(pybind_folder)
    delete_folder(os.path.join(folder, "cpu"))


def fix_open3d_stub_syntax():
    # resolve syntax issues
    path = os.path.join(os.path.dirname(__file__), "..", "typings", "open3d")
    camera = os.path.join(path, "camera", "__init__.pyi")
    with open(camera, "r") as f:
        data = f.read()
    data = data.replace("3x3 numpy array", "np.ndarray")
    data = data.replace("4x4 numpy array", "np.ndarray")
    with open(camera, "w") as f:
        f.write(data)
    with open(camera, "r") as f:
        data = f.readlines()
    data.insert(0, "import numpy as np\n")
    with open(camera, "w") as f:
        f.writelines(data)


def make_stubs(module):
    name = get_name(module)
    print(f"    Making stubs for {name}")
    subprocess.run(["stubgen", "-m", name])


def main():
    for module in [
        open3d,
        open3d.camera,
        open3d.core,
        open3d.data,
        open3d.geometry,
        open3d.io,
        open3d.ml,
        open3d.pipelines,
        open3d.t,
        open3d.utility,
        open3d.visualization,
    ]:
        print(f"Making stubs for {get_name(module)}")
        make_stubs(module)

    fix_open3d_stubs()

    move_open3d_stubs()

    fix_open3d_stub_syntax()

    delete_folder(os.path.join(os.path.dirname(__file__), "..", "out"))

if __name__ == "__main__":
    main()
