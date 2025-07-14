from flask import Blueprint, jsonify, request, g
from src.Project.Infrastructure.Repositories.EstablecimientoRepository import (
    EstablecimientoRepository,
)
from src.Project.Aplication.EstablecimientoUseCases.GetEstablecimiemto import GetEstablecimientos
from src.Project.Aplication.EstablecimientoUseCases.CreateEstablecimiento import CreateEstablecimiento
from src.Project.Aplication.EstablecimientoUseCases.DeleteEstablecimiento import DeleteEstablecimiento

from src.Project.Infrastructure.Services.EstablecimientoService import (
    EstablecimientoService,
)
from src.Project.Infrastructure.Utils.jwt_utils import token_requerido

bp_establecimiento = Blueprint("establecimiento", __name__)

# 👇 Aquí está la corrección
getter = GetEstablecimientos(EstablecimientoRepository())
creator = CreateEstablecimiento(EstablecimientoRepository())
delette= DeleteEstablecimiento(EstablecimientoRepository())
service = EstablecimientoService(getter, creator, delette)

@bp_establecimiento.route("/establecimientos", methods=["GET"])
def listar_establecimientos():
    ests = service.listar()
    return jsonify([e.__dict__ for e in ests])

@bp_establecimiento.route("/establecimientos/<int:id_>", methods=["GET"])
def obtener_establecimiento(id_):
    est = service.obtener(id_)
    return jsonify(est.__dict__) if est else ("Not Found", 404)

@bp_establecimiento.route("/establecimientos/rg", methods=["POST"])
@token_requerido
def create_establecimiento():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    try:
        nuevo_establecimiento = service.create(
            data["idalta_establecimiento"],
            data["nombre"],
            data["direccion"],
            data["ciudad"],
            data["tipo"],
            data["horario"],
            data["precio"],
            data["imagen"],
            request.id_administrador
        )
        #return jsonify(nuevo_establecimiento.__dict__), 201
        return jsonify(nuevo_establecimiento.to_dict()), 201

    except Exception as e:
        return jsonify({"error en controller": str(e)}), 500


@bp_establecimiento.route("/establecimientos/<int:id_>", methods=["DELETE"])
def eliminar_establecimiento(id_):
    est = service.delete(id_)
    if est:
        return jsonify({"mensaje": "Establecimiento eliminado correctamente", "establecimiento": est.__dict__})
    else:
        return jsonify({"error": "No se encontró el establecimiento"}), 404
