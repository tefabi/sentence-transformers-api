test:
	coverage run --source=app,models -m pytest

coverage:
	coverage report -m