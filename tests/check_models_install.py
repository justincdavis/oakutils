import os

import oakutils.blobs.models as models


def test_models_install():
    """Test that the models are installed correctly"""
    assert os.path.exists(models.__file__)
    shave_modules = [getattr(models, m) for m in dir(models) if "shave" in m]
    for shave_module in shave_modules:
        assert os.path.exists(shave_module.__file__)
        contents = [c for c in dir(shave_module)]
        for model in contents:
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
            model_path = getattr(shave_module, model)
            if "site-packages" not in model_path:
                continue
            assert os.path.exists(model_path)
