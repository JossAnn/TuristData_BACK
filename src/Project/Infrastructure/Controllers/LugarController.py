from flask import Blueprint, jsonify, request, g
from src.Project.Infrastructure.Repositories.EventosEspecialesRepository import EventosEspecialesRepository

from src.Project.Aplication.EventoEspecialesUsesCases.CreateEventosEspeciales import CreateEventosEspeciales

from src.Project.Aplication.EventoEspecialesUsesCases.GetAllEventosEpeciales import GetEventosEspecialesUseCase

from src.Project.Infrastructure.Services.EventosEspecialesService import (
    EventosEspecialesService)

from src.Project.Infrastructure.Utils.jwt_utils import token_requerido

bp_eventosespeciales = Blueprint("eventosespeciales", __name__)

# 👇 Aquí está la corrección
creator = CreateEventosEspeciales(EventosEspecialesRepository())
service = EventosEspecialesService(creator)

@bp_eventosespeciales.route("/eventosespeciales/rg", methods=['POST', 'OPTIONS'])
@token_requerido
def create_eventosEspeciales():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    try:
        # Agrega los datos adicionales que no vienen en el JSON
        data["id_lugar"] = request.json.get("id_lugar")
        data["id_temporada"] = request.id_temporada
        data["id_administrador"] = request.id_administrador

        nuevo_eventoespecial = service.create(data)
        return jsonify(nuevo_eventoespecial.to_dict()), 201

    except Exception as e:
        return jsonify({"error en controller": str(e)}), 500
