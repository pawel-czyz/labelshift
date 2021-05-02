test:
	pytype
	interrogate
	pytest

install:
	pip install -r requirements.txt
	pip install -e .

.PHONY: test
