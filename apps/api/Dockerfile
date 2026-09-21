FROM python:3.12-slim
WORKDIR /srv/api
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e .
COPY app ./app
COPY alembic ./alembic
COPY alembic.ini ./
RUN useradd -r app && chown -R app:app /srv/api
USER app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
