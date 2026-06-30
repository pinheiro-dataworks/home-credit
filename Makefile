.PHONY: install features stats train api dashboard docker-build docker-run clean

install:
	pip install -r requirements.txt
	cd dashboard && npm install

features:
	python pipeline.py --stage features

stats:
	python pipeline.py --stage stats

train:
	python pipeline.py --stage train

pipeline:
	dvc repro

api:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

api-prod:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 2

dashboard-dev:
	cd dashboard && npm run dev

dashboard-build:
	cd dashboard && npm run build

docker-build:
	docker build -t home-credit-api -f Dockerfile .

docker-run:
	docker run -p 8000:8000 home-credit-api

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf models/artifacts/*.pkl models/artifacts/*.json data/processed/*.parquet

all: install pipeline api
