.PHONY: all clean docs blobs test

all: 
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
	sphinx-apidoc -o docs/source/ oakutils/
	cd docs && make html

blobs:
	python3 scripts/compile_models.py

stubs:
	python3 scripts/make_stubs.py

test:
	python3 -m unittest discover -s tests -p '*_test.py'
