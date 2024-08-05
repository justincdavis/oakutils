# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import os

from stdlib_list import stdlib_list
import oakutils.blobs.models as models


def test_model_paths_valid():
    """Test that the models are installed correctly"""
    assert os.path.exists(models.__file__)
    stdlib = stdlib_list()
    shave_modules = [getattr(models, m) for m in dir(models) if "shave" in m]
    print(f"Found {len(shave_modules)} shave modules -> {shave_modules}")
    for shave_module in shave_modules:
        print(f"Searching {shave_module}")
        assert os.path.exists(shave_module.__file__)
        contents = [c for c in dir(shave_module)]
        for model in contents:
            # print(f"   Checking {model}")
            # print(f"       type(model) = {type(model)})")
            if model[0] == "_":
                continue
            if model in stdlib:
                continue
            if model == "os":
                continue
            if model == "sys":
                continue
            if model == "site":
                continue
            if model == "sysconfig":
                continue
            if model == "pkg_resources":
                continue
            model_path = str(getattr(shave_module, model))
            # print(f"   model_path = {model_path}")
            # print(f"       type(model_path) = {type(model_path)})")
            if "site-packages" not in model_path:
                continue
            assert os.path.exists(model_path)


def test_model_shave_dirs_equal():
    """Tests all the shave modules have the same number of models"""
    assert os.path.exists(models.__file__)
    shave_modules = [getattr(models, m) for m in dir(models) if "shave" in m]
    print(f"Found {len(shave_modules)} shave modules -> {shave_modules}")
    lengths = []
    files = []
    for shave_module in shave_modules:
        print(f"Searching {shave_module}")
        assert os.path.exists(shave_module.__file__)
        contents = [c for c in dir(shave_module) if not c.startswith("_")]
        print(f"   Found {len(contents)} models in {shave_module}")
        lengths.append(len(contents))
        files.append(contents)
    try:
        assert len(set(lengths)) == 1
    except AssertionError as err:
        # find the different file
        for idx1, module_contents1 in enumerate(files):
            for idx2, module_contents2 in enumerate(files):
                if module_contents1 == module_contents2:
                    continue
                for file in module_contents1:
                    if file not in module_contents2:
                        print(f"File {file} from shave {idx1+1} not in other module shave {idx2+1}")
        raise err

def test_model_shave_dirs_equivalent():
    """Tests all the shave modules have the same models"""
    assert os.path.exists(models.__file__)
    shave_modules = [getattr(models, m) for m in dir(models) if "shave" in m]
    print(f"Found {len(shave_modules)} shave modules -> {shave_modules}")
    contents = []
    for shave_module in shave_modules:
        print(f"Searching {shave_module}")
        assert os.path.exists(shave_module.__file__)
        contents.append([c for c in dir(shave_module) if not c.startswith("_")])
    for data in zip(*contents):
        assert len(set(data)) == 1
