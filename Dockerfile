FROM python:3.11-slim 

WORKDIR /app

# Copy file requirements.txt và cài đặt dependecies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy ứng dụng API và Feature Store SQLite vào Container
COPY ./app ./app
COPY ./ecommerce.db .

EXPOSE 8000

# Khởi chạy Uvicorn Server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]