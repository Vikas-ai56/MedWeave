# One image for every MedWeave process: the coordinator (Flower SuperLink, which runs the ServerApp
# and the relay), each institution (Flower SuperNode, which runs the ClientApp), the dashboard and
# the one-shot setup. docker-compose.yml chooses the command. See docs/docker.md.

FROM node:22-slim AS dashboard
WORKDIR /frontend
COPY dashboard/frontend/package.json dashboard/frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY dashboard/frontend/ ./
RUN npm run build

FROM python:3.14-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    MEDWEAVE_STATE_DIR=/state \
    MEDWEAVE_DATA_DIR=/data \
    FLWR_HOME=/state/flwr_home \
    GIT_PYTHON_REFRESH=quiet
RUN apt-get update && apt-get install -y --no-install-recommends openssl && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY coordinator coordinator
COPY database database
COPY institution_data institution_data
COPY privacy privacy
COPY split_learning split_learning
COPY tracking tracking
COPY dashboard/__init__.py dashboard/__init__.py
COPY dashboard/backend dashboard/backend
COPY --from=dashboard /frontend/dist dashboard/frontend/dist

# Named volumes mounted here start with this ownership, so the unprivileged user can write to them.
RUN useradd --create-home --uid 10001 medweave \
 && mkdir -p /state /data /certs \
 && chown medweave /state /data /certs
USER medweave
