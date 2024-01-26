import os

import oakutils.blobs.models as models


def test_models_install():
    """Test that the models are installed correctly"""
    assert os.path.exists(models.__file__)
    shave_modules = [getattr(models, m) for m in dir(models) if "shave" in m]
    print(f"Found {len(shave_modules)} shave modules -> {shave_modules}")
    for shave_module in shave_modules:
        print(f"Searching {shave_module}")
        assert os.path.exists(shave_module.__file__)
        contents = [c for c in dir(shave_module)]
        for model in contents:
            print(f"   Checking {model}")
            print(f"       type(model) = {type(model)})")
            if model[0] == "_":
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
            print(f"   model_path = {model_path}")
            print(f"       type(model_path) = {type(model_path)})")
            if "site-packages" not in model_path:
                continue
            assert os.path.exists(model_path)


def test_models_shave_equal():
    """Tests all the shave modules have the same number of models"""
    assert os.path.exists(models.__file__)
    shave_modules = [getattr(models, m) for m in dir(models) if "shave" in m]
    print(f"Found {len(shave_modules)} shave modules -> {shave_modules}")
    lengths = []
    for shave_module in shave_modules:
        print(f"Searching {shave_module}")
        assert os.path.exists(shave_module.__file__)
        contents = [c for c in dir(shave_module)]
        print(f"   Found {len(contents)} models in {shave_module}")
        lengths.append(len(contents))
    assert len(set(lengths)) == 1


def test_models_shaves_equivalent():
    """Tests all the shave modules have the same models"""
    assert os.path.exists(models.__file__)
    shave_modules = [getattr(models, m) for m in dir(models) if "shave" in m]
    print(f"Found {len(shave_modules)} shave modules -> {shave_modules}")
    contents = []
    for shave_module in shave_modules:
        print(f"Searching {shave_module}")
        assert os.path.exists(shave_module.__file__)
        contents.append([c for c in dir(shave_module)])
    for data in zip(*contents):
        assert len(set(data)) == 1
