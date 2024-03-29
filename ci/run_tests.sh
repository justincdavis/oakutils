#!/usr/bin/env bash

# # Set the path to the tests directory
# test_directory="./tests"

# # Loop through all Python files in the tests directory
# for file in "$test_directory"/*.py; do
#     echo "Running pytest on $file"
#     python3 -m pytest --log-cli-level=WARNING --full-trace -rP "$file"
# done

python3 -m pytest --log-cli-level=WARNING -rP tests/*
