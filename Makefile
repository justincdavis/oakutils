.PHONY: help install clean docs blobs test ci mypy

help: 
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  install    to install the package"
	@echo "  clean      to clean the directory tree"
	@echo "  docs       to generate the documentation"
	@echo "  ci 	    to run the CI workflows"
	@echo "  mypy       to run the type checker"
	@echo "  blobs      to compile the models"
	@echo "  stubs      to generate the stubs"
	@echo "  test       to run the tests"

install:
	pip3 install .

clean: 
	rm -rf build
	rm -rf dist
	rm -rf *.egg-info
	rm -rf oakutils/*.egg-info
	rm -rf src/oakutils/*.egg-info
	pyclean .

docs:
	python3 scripts/build_example_docs.py
	rm -rf docs/source/*
	sphinx-apidoc -o docs/source/ src/oakutils/
	cd docs && make html

blobs:
	python3 scripts/compile_models.py

ci: 
	-./scripts/ci/pyupgrade.sh
	python3 -m ruff ./src/oakutils --fix
	python3 -m mypy src/oakutils --config-file pyproject.toml
	python3 -m isort src/oakutils
	python3 -m black src/oakutils --safe

mypy:
	python3 -m mypy src/oakutils --config-file pyproject.toml

stubs:
	python3 scripts/make_stubs.py

test:
	./scripts/run_tests.sh
