test:
	flake8
	pytype
	interrogate
	pytest

install:
	pip install -r requirements.txt
	pip install -e .

.PHONY: test
