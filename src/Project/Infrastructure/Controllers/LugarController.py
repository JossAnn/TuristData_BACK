from flask import Blueprint, jsonify, request, g

from src.Project.Infrastructure.Repositories.LugaresRepository import LugaresRepository
from src.Project.Aplication.LugaresUseCases.CreateLugar import CreateLugares
from src.Project.Infrastructure.Services.LugaresService import LugaresService

from src.Project.Infrastructure.Utils.jwt_utils import token_requerido

bp_lugares = Blueprint("lugares", __name__)

# 👇 Aquí está la corrección
creator = CreateLugares(LugaresRepository())
service = LugaresService(creator)

@bp_lugares.route("/lugares/rg", methods=['POST', 'OPTIONS'])
@token_requerido
def create_lugar():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    try:
        # Agrega los datos adicionales que no vienen en el JSON
        data["id_lugar"] = request.json.get("id_lugar")

        nuevo_lugar = service.create(data)
        return jsonify(nuevo_lugar.to_dict()), 201

    except Exception as e:
        return jsonify({"error en controller": str(e)}), 500
