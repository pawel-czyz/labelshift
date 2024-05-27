test:
	flake8
	pytype labelshift
	pytype tests
	pytest

install:
	pip install -r requirements.txt
	pip install -e .

.PHONY: test
