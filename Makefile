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
	rm -rf docs/source/*
	sphinx-apidoc -o docs/source/ src/oakutils/
	cd docs && make html

blobs:
	python3 scripts/compile_models.py

ci:	mypy
	./scripts/ci/pyupgrade.sh
	python3 -m ruff ./src/oakutils --fix
	python3 -m isort src/oakutils
	python3 -m black src/oakutils --safe

mypy:
	./scripts/ci/mypy.sh

stubs:
	python3 scripts/make_stubs.py

test:
	python3 -m pytest -s tests/*
