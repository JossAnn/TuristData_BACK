from flask import Blueprint, jsonify, request, g
from src.Project.Infrastructure.Repositories.EventosEspecialesRepository import EventosEspecialesRepository

from src.Project.Aplication.EventoEspecialesUsesCases.CreateEventosEspeciales import CreateEventosEspeciales

from src.Project.Aplication.EventoEspecialesUsesCases.GetAllEventosEpeciales import GetEventosEspecialesUseCase

from src.Project.Infrastructure.Services.EventosEspecialesService import (
    EventosEspecialesService)

from src.Project.Infrastructure.Utils.jwt_utils import token_requerido

bp_eventosespeciales = Blueprint("eventosespeciales", __name__)

# 👇 Aquí está la corrección
getter = GetEventosEspecialesUseCase(EventosEspecialesRepository())
creator = CreateEventosEspeciales(EventosEspecialesRepository())
service = EventosEspecialesService(getter,creator)


@bp_eventosespeciales.route("/eventosespeciales", methods=["GET"])
def listar_eventosespeciales():
    ests = service.listar()
    return jsonify([e.to_dict() for e in ests])

@bp_eventosespeciales.route("/eventosespeciales/rg", methods=["POST"])
@token_requerido
def create_eventosEspeciales():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    try:
        # Agrega los datos adicionales que no vienen en el JSON
        data["id_destino"] = request.id_destino
        data["id_temporada"] = request.id_temporada
        data["id_administrador"] = request.iid_administrador

        nuevo_eventoespecial = service.create(data)
        return jsonify(nuevo_eventoespecial.to_dict()), 201

    except Exception as e:
        return jsonify({"error en controller": str(e)}), 500
