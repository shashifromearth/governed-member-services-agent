FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
COPY config ./config
COPY evaluation ./evaluation
# Editable install keeps /app as the project root (config/, evaluation/, reports/, var/).
RUN pip install --no-cache-dir -e . \
 && useradd --system --uid 10001 --home /app app \
 && mkdir -p /app/var /app/reports && chown -R app /app/var /app/reports
USER app
# The pack (kb/, eval/, ...) and every credential are supplied at run time; nothing sensitive is baked in.
ENV MSA_PACK_DIR=/pack
EXPOSE 8000
CMD ["uvicorn", "msagent.api:app", "--host", "0.0.0.0", "--port", "8000"]
