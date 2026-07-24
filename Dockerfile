FROM python:3.11-slim

WORKDIR /app

# Copy requirements first so Docker caches this layer separately from
# code changes — dependencies rarely change, code does. Keeps rebuilds
# fast when you're only tweaking main.py.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Only what the API actually needs — not the notebooks, not the Streamlit
# app, not .venv. Keeps the image small and avoids shipping dev artifacts.
COPY main.py .
COPY models/ ./models/

EXPOSE 8000

# Lets Docker/AWS know if the container is actually serving traffic, not
# just running — useful once this sits behind a load balancer or ECS.
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# 0.0.0.0 is not optional here — the container's loopback (127.0.0.1,
# uvicorn's default) is invisible from outside the container even with
# a port mapping. This is the single most common "works locally, dead
# in Docker" bug.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# Commands to run after this  : 
# docker build -t churn-api .
# docker run -p 8000:8000 churn-api