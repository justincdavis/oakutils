.PHONY: all clean blobs

all: 
	pip3 install .

clean: 
	rm -rf build
	rm -rf dist

blobs:
	python3 /models/generate.py
