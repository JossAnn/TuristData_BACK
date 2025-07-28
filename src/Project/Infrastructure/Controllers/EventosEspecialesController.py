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
    try:
        # data = {
        #     "nombre": request.form.get("nombre"),
        #     "fecha_inicio": request.form.get("fecha_inicio"),
        #     "fecha_final": request.form.get("fecha_final"),
        #     "descripcion": request.form.get("descripcion"),
        #     "estado_afectado": request.form.get("estado_afectado"),
        #     "id_lugar": request.form.get("id_lugar"),
        #     "id_temporada": request.form.get("id_temporada"),
        #     "id_administrador": g.id_administrador,  # 👈 desde JWT
        # }
        data = request.get_json()
        data["id_administrador"] = g.id_administrador  # Agregas el administrador desde el token

        nuevo_eventoespecial = service.create(data)
        return jsonify(nuevo_eventoespecial.to_dict()), 201

    except Exception as e:
        return jsonify({"error en controller": str(e)}), 500
