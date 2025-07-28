import re
import string
from unidecode import unidecode


def clean_text(text: str) -> str:
    # Pasar a minúsculas
    text = text.lower()

    # (Opcional) Eliminar tildes (si no usarás modelos preentrenados en español)
    # text = unidecode(text)

    # Eliminar puntuación
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Eliminar caracteres no alfanuméricos (excepto espacios y letras con tilde)
    text = re.sub(r"[^a-záéíóúüñ0-9\s]", "", text)

    # Eliminar múltiples espacios
    text = re.sub(r"\s+", " ", text).strip()

    return text
