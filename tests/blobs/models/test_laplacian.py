# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from functools import partial

from oakutils.blobs import get_model_path
from oakutils.nodes import get_nn_frame
from oakutils.nodes.models import create_laplacian

try:
    from basic import check_model_equivalence
    from load import create_model, run_model
    from hashs import get_run_tables, hash_file, write_model_tables
except ModuleNotFoundError:
    from .basic import check_model_equivalence
    from .load import create_model, run_model
    from .hashs import get_run_tables, hash_file, write_model_tables


def test_create() -> None:
    for ks1 in [3, 5, 7, 9, 11, 13, 15]:
        for ks2 in [3, 5, 7, 9, 11, 13, 15]:
            for shave in [1, 2, 3, 4, 5, 6]:
                for use_blur in [True, False]:
                    for use_gs in [True, False]:
                        modelfunc = partial(
                            create_laplacian,
                            kernel_size=ks1,
                            blur_kernel_size=ks2,
                            shaves=shave,
                            grayscale_out=use_gs,
                            use_blur=use_blur,
                        )
                        assert (
                            create_model(modelfunc) == 0
                        ), f"Failed for {ks1}, {ks2}, {shave}, {use_blur}, {use_gs}"


def test_run() -> None:
    hash_table, run_table = get_run_tables()
    for ks1 in [3, 5, 7, 9, 11, 13, 15]:
        for ks2 in [3, 5, 7, 9, 11, 13, 15]:
            for shave in [1, 2, 3, 4, 5, 6]:
                for use_blur in [True, False]:
                    for use_gs in [True, False]:
                        modelname = "laplacian"
                        attributes = [str(ks1)]
                        if use_blur:
                            modelname += "blur"
                            attributes.append(str(ks2))
                        if use_gs:
                            modelname += "gray"
                        modelpath = get_model_path(modelname, attributes, shave)
                        model_hash = hash_file(modelpath)
                        modelkey = modelpath.stem
                        # if the hash is the same and we have already gotten a successful run, continue
                        if hash_table[modelkey] == model_hash and run_table[modelkey]:
                            continue
                        # if the hash is not the same update the hash and set the run to false
                        existing_hash = hash_table[modelkey]
                        if existing_hash != model_hash:
                            hash_table[modelkey] = model_hash
                            run_table[modelkey] = False

                        # perform the actual run
                        modelfunc = partial(
                            create_laplacian,
                            kernel_size=ks1,
                            blur_kernel_size=ks2,
                            shaves=shave,
                            grayscale_out=use_gs,
                            use_blur=use_blur,
                        )
                        channels = 1 if use_gs else 3
                        decodefunc = partial(
                            get_nn_frame,
                            channels=channels,
                        )
                        retcode = run_model(modelfunc, decodefunc)
                        tableval = retcode == 0
                        run_table[modelkey] = tableval
                        write_model_tables(hash_table, run_table)
                        assert retcode ==0, f"Failed for {ks1}, {ks2}, {shave}, {use_blur}, {use_gs}"


def test_equivalence() -> None:
    check_model_equivalence("laplacian")


if __name__ == "__main__":
    test_create()
    test_run()
    test_equivalence()
