FROM python:3.12-slim

# Directorio de trabajo de la aplicación
WORKDIR /app

# Copiar el archivo de requerimientos
COPY requirements.txt .

# Actualizar pip e instalar dependencias Python
RUN python -m pip install --upgrade pip && \
	pip install --no-cache-dir -r requirements.txt

# ⬇️ DESCARGAR EL RECURSO NLTK 'punkt'
RUN python -m nltk.downloader punkt

# Copiar el resto de la aplicación
COPY . .

# Comando por defecto para ejecutar la aplicación
CMD ["python", "-u", "app.py"]
