from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
import os
from src.Project.Infrastructure.Utils.jwt_utils import verificar_token

from flask import g

bp_upload = Blueprint("upload", __name__)

UPLOAD_FOLDER = "uploads"

@bp_upload.route('/upload-image', methods=['POST'])
def upload_image():
    auth_header = g.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return jsonify({"error": "Token no proporcionado"}), 401

    token = auth_header.split(" ")[1]

    try:
        payload = verificar_token(token)
        id_administrador = payload.get("sub")
    except Exception as e:
        return jsonify({"error": str(e)}), 401

    image = request.files.get('imagen')
    if not image:
        return jsonify({"error": "No se envió imagen"}), 400

    filename = secure_filename(image.filename)
    ruta_carpeta = os.path.join("uploads", f"admin_{id_administrador}")
    os.makedirs(ruta_carpeta, exist_ok=True)

    ruta_final = os.path.join(ruta_carpeta, filename)
    image.save(ruta_final)

    url = f"http://localhost:8000/uploads/admin_{id_administrador}/{filename}"
    return jsonify({"url": url}), 201