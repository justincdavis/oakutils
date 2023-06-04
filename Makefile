.PHONY: all clean docs blobs

all: 
	pip3 install .

clean: 
	rm -rf build
	rm -rf dist
	rm -rf *.egg-info
	rm -rf src/*.egg-info

docs:
	rm -rf docs/source/*
	sphinx-apidoc -o docs/source/ src/
	cd docs && make html

blobs:
	python3 scripts/compile_models.py

stubs:
	python3 scripts/make_stubs.py
