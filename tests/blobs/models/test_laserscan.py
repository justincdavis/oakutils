# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from functools import partial

from oakutils.blobs import get_model_path
from oakutils.nodes.models import create_laserscan, get_laserscan

try:
    from basic import check_model_equivalence
    from load import create_model, run_model
    from hashs import get_run_tables, write_model_tables, hash_file
except ModuleNotFoundError:
    from .basic import check_model_equivalence
    from .load import create_model, run_model
    from .hashs import get_run_tables, write_model_tables, hash_file


def test_create() -> None:
    for width in [5, 10, 20]:
        for scan in [1, 3, 5]:
            for shave in [1, 2, 3, 4, 5, 6]:
                modelfunc = partial(
                    create_laserscan,
                    width=width,
                    scans=scan,
                    shaves=shave,
                )
                assert (
                    create_model(modelfunc) == 0
                ), f"Failed for {width}, {scan}, {shave}"


def test_run() -> None:
    hash_table, run_table = get_run_tables()
    for width in [5, 10, 20]:
        for scan in [1, 3, 5]:
            for shave in [1, 2, 3, 4, 5, 6]:
                modelname = "laserscan"
                attributes = [str(width), str(scan)]
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

                modelfunc = partial(
                    create_laserscan,
                    width=width,
                    scans=scan,
                    shaves=shave,
                )
                decodefunc = get_laserscan
                retcode = run_model(modelfunc, decodefunc)
                tableval = retcode == 0
                run_table[modelkey] = tableval
                write_model_tables(hash_table, run_table)
                assert retcode == 0, f"Failed for {width}, {scan}, {shave}"


def test_equivalence() -> None:
    check_model_equivalence("laserscan")


if __name__ == "__main__":
    test_create()
    test_run()
    test_equivalence()
