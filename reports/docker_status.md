# Docker status (2026-11-10)

| Check | Status | Evidence |
|---|---|---|
| Dockerfile written | done | `Dockerfile`, `.dockerignore`, `requirements-api.txt` |
| Linux wheels resolve for every pinned API dependency (cp312, manylinux) | done | `pip download --platform manylinux… --only-binary=:all:` succeeded for numpy 2.5.0, scipy 1.18.0, scikit-learn 1.9.0, pydantic-core, fastapi 0.162.2, uvicorn 0.54.0 |
| Same API code runs under uvicorn outside Docker | done | `reports/api_run_evidence.json` |
| `docker build` | **not run** | Docker is not installed on the development machine |
| `docker run` + `/health` + valid/invalid `/predict` | **not run** | same |

To close this item after installing Docker Desktop:

```bash
docker build -t robot-ad-api:1.0.0 .
docker run --rm -p 8000:8000 robot-ad-api:1.0.0
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/predict -H "content-type: application/json" --data @data/examples/test_normal_1963.json
curl -X POST http://127.0.0.1:8000/predict -H "content-type: application/json" --data @data/examples/invalid/missing_channel.json
```

Record the build log, image size, the three responses and the container log
here, then mark D06 Docker as passed.
