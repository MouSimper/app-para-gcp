# 1. Imagen base de NVIDIA con CUDA 11.8
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# 2. Instalar Python 3.10 y pip
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    && apt-get clean

# 3. Hacer que python3.10 sea el 'python' por defecto
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# 4. Crear directorio y copiar dependencias
WORKDIR /app
COPY requirements.txt .

# 5. Instalar dependencias
# bitsandbytes requiere esta bandera especial
RUN CMAKE_ARGS="-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc" pip install --no-cache-dir -r requirements.txt

# 6. Copiar el resto de tus archivos de código
# (El script .py, el label_encoder.pkl, y el logo.png)
COPY . .

# 7. Exponer el puerto y correr la app
EXPOSE 8080
CMD ["python", "chatbot_llama3_4bit_con_clasificador.py"]