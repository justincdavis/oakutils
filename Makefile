.PHONY: all clean

all: 
	pip3 install .

clean: 
	rm -rf build
	rm -rf dist
