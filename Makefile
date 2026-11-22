.PHONY: qdrant samples index eval eval-scale test run docker docker-gpu
qdrant:       ; docker run -d --name cardagent-qdrant -p 127.0.0.1:6333:6333 -v cardagent_qdrant:/qdrant/storage qdrant/qdrant:v1.18.1
samples:      ; python tools/make_samples.py
index:        ; python tools/build_index.py
eval:         ; python tools/evaluate.py --today $$(date +%F)
eval-scale:   ; python tools/evaluate.py --manifest data/samples_scale/manifest.json --out data/eval_scale_report.json
test:         ; pytest -q
run:          ; PYTHONPATH=src uvicorn cardagent.api.app:app --host 0.0.0.0 --port 8080
docker:       ; docker build -f deploy/Dockerfile -t cardagent:1.0.0 .
docker-gpu:   ; docker build -f deploy/Dockerfile.gpu -t cardagent:1.0.0-gpu .
