.PHONY: help install clean docs blobs test ci mypy pyright pyupgrade isort black ruff release example-ci

help: 
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  install    to install the package"
	@echo "  clean      to clean the directory tree"
	@echo "  docs       to generate the documentation"
	@echo "  ci 	    to run the CI workflows"
	@echo "  mypy       to run the mypy static type checker"
	@echo "  pyright    to run the pyright static type checker"
	@echo "  pyupgrade  to run pyupgrade"
	@echo "  isort      to run isort"
	@echo "  black      to run black"
	@echo "  ruff 	    to run ruff"
	@echo "  blobs      to compile the models"
	@echo "  stubs      to generate the stubs"
	@echo "  test       to run the tests"
	@echo "  release    to perform all actions required for a release"
	@echo "  example-ci to run the CI workflows for the example scripts"

install:
	pip3 install .

clean: 
	rm -rf build
	rm -rf dist
	rm -rf *.egg-info
	rm -rf oakutils/*.egg-info
	rm -rf src/oakutils/*.egg-info
	pyclean .
	rm -rf .mypy_cache
	rm -rf .ruff_cache

docs:
	python3 ci/build_example_docs.py
	rm -rf docs/source/*
	sphinx-apidoc -o docs/source/ src/oakutils/
	cd docs && make html

blobs:
	python3 ci/compile_models.py --definitions

ci: pyupgrade ruff isort black mypy

mypy:
	python3 -m mypy src/oakutils --config-file=pyproject.toml

pyright:
	python3 -m pyright --project=pyproject.toml

pyupgrade:
	-./ci/pyupgrade.sh

isort:
	python3 -m isort src/oakutils

black:
	python3 -m black src/oakutils --safe

ruff:
	python3 -m ruff check ./src/oakutils --fix --preview

stubs:
	python3 ci/make_stubs.py

test:
	./ci/run_tests.sh

example-ci: pyupgrade
	python3 -m ruff ./examples --fix --preview --ignore=T201,INP001,F841
	python3 -m isort examples
	python3 -m black examples --safe

release: clean blobs ci test docs example-ci
