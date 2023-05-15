.PHONY: all clean docs

all: 
	pip3 install .

clean: 
	rm -rf build
	rm -rf dist

docs:
	rm -rf docs/source/*
	sphinx-apidoc -o docs/source/ src/
	cd docs && make html
