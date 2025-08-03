FROM python:3.12-slim

# Directorio de trabajo de la aplicación
WORKDIR /app

# Copiar el archivo de requerimientos
COPY requirements.txt .

# Actualizar pip e instalar dependencias Python
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Descargar recursos de NLTK durante la construcción del contenedor
RUN python -c "import nltk; import ssl; \
    try: \
        _create_unverified_https_context = ssl._create_unverified_context; \
    except AttributeError: \
        pass; \
    else: \
        ssl._create_default_https_context = _create_unverified_https_context; \
    print('Descargando recursos de NLTK...'); \
    nltk.download('punkt_tab', quiet=True); \
    nltk.download('punkt', quiet=True); \
    print('Recursos NLTK descargados exitosamente')"

# Copiar el resto de la aplicación
COPY . .

# Comando por defecto para ejecutar la aplicación
CMD ["python", "-u", "app.py"]