# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import hashlib
import pickle
import io
from pathlib import Path

from oakutils.blobs.models.bulk import ALL_MODELS


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
            hash_table[blob_path] = hash_file(blob_path)
    table_file = Path(__file__).parent / "hash_table.pkl"
    with Path.open(table_file, "wb") as file:
        pickle.dump(hash_table, file, protocol=pickle.HIGHEST_PROTOCOL)


def compare_entry(entry: Path) -> bool:
    with Path.open(Path(__file__).parent / "hash_table.pkl", "rb") as file:
        table = pickle.load(file)
    return table[entry] == hash_file(entry)


def create_bulk_hash_table() -> None:
    hash_table: dict[str, str] = {}
    for blob_tuple in ALL_MODELS:
        # get the stem file path without the suffix
        # then remove the _shavesN part at the end
        key = blob_tuple[0].stem[:-8]
        hashes = [hash_file(bp) for bp in blob_tuple]
        hash_table[key] = hash(tuple(hashes))
    table_file = Path(__file__).parent / "bulk_hash_table.pkl"
    with Path.open(table_file, "wb") as file:
        pickle.dump(hash_table, file, protocol=pickle.HIGHEST_PROTOCOL)


def compare_bulk_entry(entry: tuple[Path, ...]) -> bool:
    key = entry[0].stem[:-8]
    with Path.open(Path(__file__).parent / "bulk_hash_table.pkl", "rb") as file:
        table = pickle.load(file)
    return hash(tuple(hash_file(bp) for bp in entry)) == table[key]
