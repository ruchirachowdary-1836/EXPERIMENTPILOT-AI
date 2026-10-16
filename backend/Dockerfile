FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
WORKDIR /srv
COPY engine ./engine
COPY backend ./backend
COPY research/results ./research/results
RUN pip install --no-cache-dir ./engine ./backend
WORKDIR /srv/backend
RUN useradd -m app && mkdir -p /srv/backend/data && chown -R app /srv/backend/data
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
