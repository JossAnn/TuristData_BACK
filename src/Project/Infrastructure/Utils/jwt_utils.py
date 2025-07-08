import jwt
import datetime
from dotenv import load_dotenv
import os
from functools import wraps
from flask import request, jsonify, g

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

def verificar_token(token):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise ValueError("Token expirado")
    except jwt.InvalidTokenError:
        raise ValueError("Token inválido")

def token_requerido(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        #auth_header = request.headers.get("Authorization")
        auth_header = g.get("Authorization")
        
        if not auth_header or not auth_header.startswith("Bearer "):
            return jsonify({"error": "Token no proporcionado"}), 401

        token = auth_header.split(" ")[1]

        try:
            payload = verificar_token(token)
            g.id_administrador = payload.get("sub")   # guardamos el id_administrador en request
        except ValueError as e:
            return jsonify({"error jwt utils": str(e)}), 401

        return f(*args, **kwargs)

    return decorated