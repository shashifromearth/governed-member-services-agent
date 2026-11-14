FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
COPY config ./config
COPY evaluation ./evaluation
RUN pip install --no-cache-dir . && useradd --system --uid 10001 app && mkdir /app/var && chown app /app/var
USER app
# The pack (kb/) and all credentials are supplied at run time; nothing sensitive is baked in.
ENV MSA_PACK_DIR=/pack MSA_VAR_DIR=/app/var MSA_TENANTS_FILE=/app/config/tenants.json
EXPOSE 8000
CMD ["uvicorn", "msagent.api:app", "--host", "0.0.0.0", "--port", "8000"]
