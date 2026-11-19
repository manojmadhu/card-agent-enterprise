.PHONY: samples index eval eval-scale test run docker docker-gpu
samples:      ; python tools/make_samples.py
index:        ; python tools/build_index.py --force
eval:         ; python tools/evaluate.py --today $$(date +%F)
eval-scale:   ; python tools/evaluate.py --manifest data/samples_scale/manifest.json --out data/eval_scale_report.json
test:         ; pytest -q
run:          ; PYTHONPATH=src uvicorn cardagent.api.app:app --host 0.0.0.0 --port 8080
docker:       ; docker build -f deploy/Dockerfile -t cardagent:1.0.0 .
docker-gpu:   ; docker build -f deploy/Dockerfile.gpu -t cardagent:1.0.0-gpu .
