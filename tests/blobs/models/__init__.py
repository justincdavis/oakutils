# Copyright (c) 2024 Justin Davis (davisjustin302@gmail.com)
#
# MIT License
from __future__ import annotations

from pathlib import Path

from .hashs import create_file_hash_table, create_bulk_hash_table

# handle the creation of the hash files if they do not exists
hash_table_path = Path(__file__).parent / "hash_table.pkl"
if not hash_table_path.exists():
    create_file_hash_table()

bulk_hash_table_path = Path(__file__).parent / "bulk_hash_table.pkl"
if not bulk_hash_table_path.exists():
    create_bulk_hash_table()
