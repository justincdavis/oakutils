# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

import pickle
from pathlib import Path

from .hashs import (
    create_file_hash_table,
    create_bulk_hash_table,
    HASH_TABLE_PATH,
    MODEL_RUN_TABLE_PATH,
    BULK_HASH_TABLE_PATH,
    BULK_RUN_TABLE_PATH,
)

# handle the creation of the hash files if they do not exists
# hash table for model to hash and model to successful run
if not HASH_TABLE_PATH.exists():
    create_file_hash_table()
if not MODEL_RUN_TABLE_PATH.exists():
    model_run_table = {}
    with Path.open(HASH_TABLE_PATH, "rb") as file:
        hash_table = pickle.load(file)
    for key in hash_table:
        model_run_table[key] = False
    with Path.open(MODEL_RUN_TABLE_PATH, "wb") as file:
        pickle.dump(model_run_table, file, protocol=pickle.HIGHEST_PROTOCOL)

# hash table for bulk model to hash and bulk model to successful run
if not BULK_HASH_TABLE_PATH.exists():
    create_bulk_hash_table()
if not BULK_RUN_TABLE_PATH.exists():
    bulk_run_table = {}
    with Path.open(BULK_HASH_TABLE_PATH, "rb") as file:
        bulk_hash_table = pickle.load(file)
    for key in bulk_hash_table:
        bulk_run_table[key] = False
    with Path.open(BULK_RUN_TABLE_PATH, "wb") as file:
        pickle.dump(bulk_run_table, file, protocol=pickle.HIGHEST_PROTOCOL)
