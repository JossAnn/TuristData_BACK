import jwt
import datetime
from dotenv import load_dotenv
import os

load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEYJWT")

def crear_token(admin_id, correo):
    payload = {
        "sub": admin_id,
        "correo": correo,
        "iat": datetime.datetime.utcnow(),
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=1)  # token válido 1 hora
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
    return token
