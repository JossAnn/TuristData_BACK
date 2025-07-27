from flask import Blueprint, jsonify, request, g

from src.Project.Infrastructure.Repositories.ComentarioRepository import ComentarioRepository
from src.Project.Aplication.ComentarioUseCases.CreateComentario import CreateComentario
from src.Project.Aplication.ComentarioUseCases.GetComentario import GetComentario
from src.Project.Infrastructure.Services.ComentarioService import ComentarioService

from src.Project.Infrastructure.Repositories.EstablecimientoRepository import EstablecimientoRepository
establecimiento_repo = EstablecimientoRepository()

from src.Project.Infrastructure.Utils.jwt_utils import token_requerido

bp_comentario = Blueprint("comentarios", __name__)

# 👇 Aquí está la corrección
getter = GetComentario(ComentarioRepository())
creator = CreateComentario(ComentarioRepository())
service = ComentarioService(getter,creator)


@bp_comentario.route("/comentario", methods=["GET"])
def listar_comentario():
    ests = service.listar()
    return jsonify([e.to_dict() for e in ests])


@bp_comentario.route("/comentario/establecimiento", methods=["GET"])
def listar_comentario_establecimiento():
    id_establecimiento = request.args.get("establecimiento_id", type=int)

    if not id_establecimiento:
        return jsonify({"error": "Debe especificar id del establecimiento"}), 400

    comentarios = service.get_by_establecimiento(id_establecimiento)

    return jsonify(comentarios)


@bp_comentario.route("/comentario/rg", methods=["POST"])
@token_requerido
def crear_comentario():
    try:
        data = request.json

        comentario = data.get("comentario")
        try:
            estrellas_calificacion = int(data.get("estrellas_calificacion"))
        except (TypeError, ValueError):
            return jsonify({"error": "La calificación debe ser un número entero"}), 400

        id_establecimiento = data.get("id_establecimiento")
        id_turista = g.id_usuario

        if not comentario or not id_establecimiento:
            return jsonify({"error": "Faltan datos necesarios"}), 400

        comentario_data = {
            "comentario": comentario,
            "estrellas_calificacion": estrellas_calificacion,
            "id_usuario": id_turista,
            "idalta_establecimiento": id_establecimiento
        }

        nuevo_comentario = service.create(comentario_data)
        return jsonify(nuevo_comentario), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500

