# 1. Imagen base de Python 3.10 (NO nvidia/cuda)
FROM python:3.10-slim

# 2. Instalar dependencias del sistema (para 'torch' y 'scikit-learn')
# build-essential es necesario para compilar algunos paquetes de pip
RUN apt-get update && apt-get install -y \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 3. Crear directorio de trabajo y copiar dependencias
WORKDIR /app
COPY requirements.txt .

# 4. Instalar dependencias de Python
# (Pip instalará la versión de CPU de PyTorch automáticamente)
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copiar el resto de la app
# (app.py, label_encoder.pkl, logo.png)
COPY . .

# 6. Exponer el puerto que Google Cloud espera
EXPOSE 8080

# 7. ¡IMPORTANTE! Ejecutar "app.py" (el script de Gemini),
# NO el script antiguo.
CMD ["python", "app.py"]