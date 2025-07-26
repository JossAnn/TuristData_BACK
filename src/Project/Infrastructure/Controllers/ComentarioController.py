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



@bp_comentario.route("/comentario/rg", methods=["POST"])
@token_requerido
def create_comentario():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    try:
        data["id_usuario"] = g.id_usuario

        if "idalta_establecimiento" not in data:
            return jsonify({"error": "Debe enviar idalta_establecimiento"}), 400

        id_establecimiento = data["idalta_establecimiento"]
        id_administrador = establecimiento_repo.obtener_administrador_por_establecimiento(id_establecimiento)

        if not id_administrador:
            return jsonify({"error": "No se encontró administrador para ese establecimiento"}), 400

        data["id_administrador"] = id_administrador

        nuevo_comentario = service.create(data)
        return jsonify(nuevo_comentario.to_dict()), 201

    except Exception as e:
        return jsonify({"error en controller": str(e)}), 500

