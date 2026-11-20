.PHONY: scrape index test run docker eval
scrape:
	python -m ingest.scrape_darglobal
	python -m ingest.scrape_wasalt --max-listings 1500
index:
	python -m ingest.build_index
test:
	pytest -q
run:
	uvicorn app.main:app --reload --port 8000
docker:
	docker build -t darglobal-chatbot . && docker run --rm -p 8000:10000 -m 512m -e OPENROUTER_API_KEY=$$OPENROUTER_API_KEY darglobal-chatbot
eval:
	python scripts/eval.py --url http://localhost:8000
