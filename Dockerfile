# Imagen base: Python 3.13 (misma versión usada para entrenar en Colab)
FROM python:3.13-slim

# Evita archivos .pyc y muestra los logs en tiempo real
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# 1) Primero solo las dependencias (aprovecha la caché de capas de Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2) Luego el resto del proyecto (app.py, model.joblib, train.py, data/...)
COPY . .

# Puerto en el que escucha Streamlit
EXPOSE 8501

# Docker revisa periódicamente que la app responda
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

# Proceso que se ejecuta al iniciar el contenedor
CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true"]
