FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py model.keras model_config.json ./
COPY examples ./examples
EXPOSE 7860
CMD ["python", "app.py"]
