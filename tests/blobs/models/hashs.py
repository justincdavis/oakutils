# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import hashlib
import pickle
import io
from pathlib import Path

from oakutils.blobs.models.bulk import ALL_MODELS


HASH_TABLE_PATH = Path(__file__).parent / "cache" / "hash_table.pkl"
MODEL_RUN_TABLE_PATH = Path(__file__).parent / "cache" / "model_run_table.pkl"
BULK_HASH_TABLE_PATH = Path(__file__).parent / "cache" / "bulk_hash_table.pkl"
BULK_RUN_TABLE_PATH = Path(__file__).parent / "cache" / "bulk_run_table.pkl"


def hash_file(file_path: Path) -> str:
    hasher = hashlib.md5()
    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(io.DEFAULT_BUFFER_SIZE), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def create_file_hash_table() -> None:
    hash_table: dict[str, str] = {}
    for blob_tuple in ALL_MODELS:
        for blob_path in blob_tuple:
            hash_table[blob_path.stem] = hash_file(blob_path)
    with Path.open(HASH_TABLE_PATH, "wb") as file:
        pickle.dump(hash_table, file, protocol=pickle.HIGHEST_PROTOCOL)


def compare_entry(entry: Path) -> bool:
    with Path.open(HASH_TABLE_PATH, "rb") as file:
        table = pickle.load(file)
    return table[entry] == hash_file(entry)


def hash_bulk_entry(entry: tuple[Path, ...]) -> str:
    hashes = [hash_file(bp) for bp in sorted(entry)]
    hashstr = "".join(hashes)
    return hashlib.md5(hashstr.encode()).hexdigest()


def create_bulk_hash_table() -> None:
    hash_table: dict[str, int] = {}
    for blob_tuple in ALL_MODELS:
        # get the stem file path without the suffix
        # then remove the _shavesN part at the end
        key = blob_tuple[0].stem[:-8]
        hash_table[key] = hash_bulk_entry(blob_tuple)
    with Path.open(BULK_HASH_TABLE_PATH, "wb") as file:
        pickle.dump(hash_table, file, protocol=pickle.HIGHEST_PROTOCOL)


def compare_bulk_entry(entry: tuple[Path, ...]) -> bool:
    key = entry[0].stem[:-8]
    with Path.open(BULK_HASH_TABLE_PATH, "rb") as file:
        table = pickle.load(file)
    return hash(tuple(hash_file(bp) for bp in entry)) == table[key]


def get_run_tables() -> tuple[dict[str, str], dict[str, bool]]:
    with Path.open(HASH_TABLE_PATH, "rb") as file:
        hash_table = pickle.load(file)
    with Path.open(MODEL_RUN_TABLE_PATH, "rb") as file:
        model_run_table = pickle.load(file)
    return hash_table, model_run_table


def write_model_tables(hash_table: dict[str, str], model_run_table: dict[str, bool]) -> None:
    with Path.open(HASH_TABLE_PATH, "wb") as file:
        pickle.dump(hash_table, file, protocol=pickle.HIGHEST_PROTOCOL)
    with Path.open(MODEL_RUN_TABLE_PATH, "wb") as file:
        pickle.dump(model_run_table, file, protocol=pickle.HIGHEST_PROTOCOL)


def get_bulk_tables() -> tuple[dict[str, int], dict[str, bool]]:
    with Path.open(BULK_HASH_TABLE_PATH, "rb") as file:
        hash_table = pickle.load(file)
    with Path.open(BULK_RUN_TABLE_PATH, "rb") as file:
        run_table = pickle.load(file)
    return hash_table, run_table


def write_bulk_tables(hash_table: dict[str, int], run_table: dict[str, bool]) -> None:
    with Path.open(BULK_HASH_TABLE_PATH, "wb") as file:
        pickle.dump(hash_table, file, protocol=pickle.HIGHEST_PROTOCOL)
    with Path.open(BULK_RUN_TABLE_PATH, "wb") as file:
        pickle.dump(run_table, file, protocol=pickle.HIGHEST_PROTOCOL)
