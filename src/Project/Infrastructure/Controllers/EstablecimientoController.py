from flask import Blueprint, jsonify, request
from src.Project.Infrastructure.Repositories.EstablecimientoRepository import (
    EstablecimientoRepository,
)
from src.Project.Aplication.EstablecimientoUseCases.GetEstablecimiemto import GetEstablecimientos
from src.Project.Infrastructure.Services.EstablecimientoService import (
    EstablecimientoService,
)

bp_establecimiento = Blueprint("establecimiento", __name__)
service = EstablecimientoService(GetEstablecimientos(EstablecimientoRepository()))


@bp_establecimiento.route("/establecimientos", methods=["GET"])
def listar_establecimientos():
    ests = service.listar()
    return jsonify([e.__dict__ for e in ests])


@bp_establecimiento.route("/establecimientos/<int:id_>", methods=["GET"])
def obtener_establecimiento(id_):
    est = service.obtener(id_)
    return jsonify(est.__dict__) if est else ("Not Found", 404)
