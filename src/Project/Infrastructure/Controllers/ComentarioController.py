from flask import Blueprint, jsonify, request, g

from src.Project.Infrastructure.Repositories.ComentarioRepository import ComentarioRepository
from src.Project.Aplication.ComentarioUseCases.CreateComentario import CreateComentario
from src.Project.Aplication.ComentarioUseCases.GetComentario import GetComentario
from src.Project.Infrastructure.Services.ComentarioService import ComentarioService

from src.Project.Infrastructure.Utils.jwt_utils import token_requerido

bp_comentario = Blueprint("comentarios", __name__)

# 👇 Aquí está la corrección
getter = GetComentario(ComentarioRepository())
creator = CreateComentario(ComentarioRepository())
service = ComentarioService(getter,creator)


@bp_comentario.route("/comentario", methods=["GET"])
def listar_destinos():
    ests = service.listar()
    return jsonify([e.to_dict() for e in ests])

@bp_comentario.route("/comentario/rg", methods=["POST"])
@token_requerido
def create_destino():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    try:
        # Agrega los datos adicionales que no vienen en el JSON
        data["id_usuario"] = request.id_usuario
        data["idalta_establecimiento"] = request.id_establecimiento
        data["id_administrador"] = request.id_administrador
        nuevo_destino = service.create(data)
        return jsonify(nuevo_destino.to_dict()), 201

    except Exception as e:
        return jsonify({"error en controller": str(e)}), 500
