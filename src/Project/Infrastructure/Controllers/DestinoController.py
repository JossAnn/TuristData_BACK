from flask import Blueprint, jsonify, request, g


from src.Project.Infrastructure.Repositories.DestinoRepository import DestinosRepository
from src.Project.Aplication.DestinoUseCases.CreateDesinos import CreateDestinos
from src.Project.Aplication.DestinoUseCases.GetDestinos import GetDestinos
from src.Project.Infrastructure.Services.DestinoService import DestinosService


from src.Project.Infrastructure.Utils.jwt_utils import token_requerido

bp_destinos = Blueprint("destinos", __name__)

# 👇 Aquí está la corrección
getter = GetDestinos(DestinosRepository())
creator = CreateDestinos(DestinosRepository())
service = DestinosService(getter,creator)


@bp_destinos.route("/destinos", methods=["GET"])
def listar_destinos():
    ests = service.listar()
    return jsonify([e.to_dict() for e in ests])

@bp_destinos.route("/destinos/rg", methods=["POST"])
@token_requerido
def create_destino():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    try:
        # Agrega los datos adicionales que no vienen en el JSON
        data["id_turista"] = request.id_turista
        nuevo_destino = service.create(data)
        return jsonify(nuevo_destino.to_dict()), 201

    except Exception as e:
        return jsonify({"error en controller": str(e)}), 500
