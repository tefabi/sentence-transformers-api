test:
	coverage run --source=app,models -m pytest -s

coverage:
	coverage report -m