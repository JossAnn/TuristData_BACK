from flask import Blueprint, jsonify, request, g
from src.Project.Infrastructure.Repositories.EventosEspecialesRepository import EventosEspecialesRepository

from src.Project.Aplication.EventoEspecialesUsesCases.CreateEventosEspeciales import CreateEventosEspeciales

from src.Project.Infrastructure.Services.EventosEspecialesService import (
    EventosEspecialesService)

from src.Project.Infrastructure.Utils.jwt_utils import token_requerido

bp_eventosespeciales = Blueprint("eventosespeciales", __name__)

# 👇 Aquí está la corrección
creator = CreateEventosEspeciales(EventosEspecialesRepository())
service = EventosEspecialesService(creator)

@bp_eventosespeciales.route("/eventosespeciales/rg", methods=["POST"])
@token_requerido
def create_eventosEspeciales():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    try:
        nuevo_eventoespecial = service.create(
            data["nombre"],
            data["fecha_inicio"],
            data["fecha_final"],
            data["descripcion"],
            data["estado_afectado"],
            request.id_destino,
            request.id_temporada,
            request.iid_administrador
        )
        #return jsonify(nuevo_establecimiento.__dict__), 201
        return jsonify(nuevo_eventoespecial.to_dict()), 201

    except Exception as e:
        return jsonify({"error en controller": str(e)}), 500
