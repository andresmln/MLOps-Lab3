# --- ETAPA 1: Base ---
# Usamos Python 3.11 para compatibilidad con PyTorch/ONNX
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# --- ETAPA 2: Builder ---
FROM base AS builder

# Instalamos uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copiamos la configuración de dependencias
COPY pyproject.toml uv.lock ./

# Instalamos las librerías (incluyendo torch, onnx, etc.) en el sistema
RUN uv export --frozen --no-dev --format requirements-txt > requirements.txt && \
    pip install --no-cache-dir --prefix=/usr/local -r requirements.txt

# --- ETAPA 3: Runtime ---
FROM base AS runtime

# Copiamos las librerías instaladas
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copiamos el código de la aplicación
COPY api ./api
COPY mylib ./mylib
COPY templates ./templates

# --- IMPORTANTE LAB 3: Copiamos el modelo y las etiquetas ---
COPY model.onnx .
COPY classes.json .

# Exponemos el puerto
EXPOSE 8000

# Arrancamos la API
CMD ["uvicorn", "api.api:app", "--host", "0.0.0.0", "--port", "8000"]
