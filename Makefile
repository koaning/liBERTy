install:
	python -m pip install -r requirements-lite.txt

benchmark:
	python benchmark-sgd.py
	python benchmark-sklearn.py
