# API image. Serves a bundle that was trained beforehand by the pipeline;
# the container never retrains and never needs the raw dataset.
#
#   docker build -t robot-ad-api:1.0.0 .
#   docker run --rm -p 8000:8000 robot-ad-api:1.0.0
#
# NOTE: not yet built on the development machine (Docker is not installed
# there); see README "Docker" and reports/docker_status.md.
FROM python:3.12.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src:/app \
    ROBOT_AD_BUNDLE=/app/models/iforest__full_18.joblib

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src/ src/
COPY api/ api/
COPY configs/ configs/
# Bundle must exist locally: run scripts/01-04 first.
COPY models/iforest__full_18.joblib models/iforest__full_18.json models/

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
