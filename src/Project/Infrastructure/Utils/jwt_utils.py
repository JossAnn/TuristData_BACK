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
        print("=== DEBUG JWT ===")
        print(f"URL: {request.url}")
        print(f"Method: {request.method}")
        print(f"Headers completos: {dict(request.headers)}")
        
        # Verificar si existe el header Authorization
        auth_header = request.headers.get("Authorization")
        print(f"Authorization header encontrado: {auth_header}")
        
        if not auth_header:
            print("ERROR: Header Authorization no encontrado")
            return jsonify({"error": "Token no proporcionado - header Authorization faltante"}), 401
        
        print(f"Longitud del header: {len(auth_header)}")
        print(f"Inicia con 'Bearer ': {auth_header.startswith('Bearer ')}")
        
        if not auth_header.startswith("Bearer "):
            print(f"ERROR: Header no inicia con 'Bearer ': {repr(auth_header)}")
            return jsonify({"error": "Token no proporcionado - formato incorrecto"}), 401

        # Extraer el token
        try:
            token = auth_header.split(" ")[1]
            print(f"Token extraído (primeros 50 chars): {token[:50]}...")
            print(f"Longitud del token: {len(token)}")
        except IndexError:
            print("ERROR: No se pudo extraer el token del header")
            return jsonify({"error": "Token no proporcionado - error al extraer token"}), 401
        
        if not token or token.strip() == "":
            print("ERROR: Token vacío")
            return jsonify({"error": "Token vacío"}), 401

        # Verificar el token
        try:
            print("Verificando token...")
            payload = verificar_token(token)
            print(f"Token válido. Payload: {payload}")
            request.id_administrador = payload.get("sub")
            print(f"ID administrador asignado: {request.id_administrador}")
        except ValueError as e:
            print(f"ERROR al verificar token: {str(e)}")
            return jsonify({"error jwt utils": str(e)}), 401

        print("=== FIN DEBUG JWT ===")
        return f(*args, **kwargs)

    return decorated