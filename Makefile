.PHONY: qdrant validate sync run test eval docker docker-gpu
# Qdrant on this machine (data kept in the docker volume cardagent_qdrant)
qdrant:   ; docker run -d --name cardagent-qdrant -p 127.0.0.1:6333:6333 -v cardagent_qdrant:/qdrant/storage qdrant/qdrant:v1.18.1
validate: ; python tools/sync_registry.py --validate-only
sync:     ; python tools/sync_registry.py
run:      ; PYTHONPATH=src uvicorn cardagent.api.app:app --host 127.0.0.1 --port 8080
test:     ; pytest -q
# make eval MANIFEST=data/eval/manifest.json
eval:     ; python tools/evaluate.py --manifest $(MANIFEST) --today $$(date +%F)
docker:   ; docker build -f deploy/Dockerfile -t cardagent:1.0.0 .
docker-gpu: ; docker build -f deploy/Dockerfile.gpu -t cardagent:1.0.0-gpu .
