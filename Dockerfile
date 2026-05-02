# Sử dụng Python 3.10
FROM python:3.10-slim

# Thiết lập thư mục làm việc
WORKDIR /app

# Khắc phục lỗi libGL cho OpenCV (cv2)
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy file requirements và cài đặt dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ mã nguồn vào container
COPY . .

# Tạo các thư mục cần thiết
RUN mkdir -p uploads outputs data

# Mở cổng 5000 cho Flask
EXPOSE 5000

# Lệnh chạy ứng dụng
CMD ["python", "app.py"]
