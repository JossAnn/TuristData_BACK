from flask import Blueprint, jsonify, request
from src.Project.Infrastructure.Repositories.TemporadaRepository import TemporadaRepository
from src.Project.Aplication.TemporadaUseCases.CreateTemporada import CreateTemporada
from src.Project.Infrastructure.Services.TemporadaService import TemporadaService
from src.Project.Infrastructure.Utils.jwt_utils import token_requerido
from flask import Blueprint, jsonify, request, g

bp_temporadas = Blueprint("temporadas", __name__)

creator = CreateTemporada(TemporadaRepository())
service = TemporadaService(creator)

@bp_temporadas.route("/temporada/rg", methods=["POST"])
@token_requerido
def create_temporada():
    data = request.get_json()

    campos_requeridos = ["nombre", "fecha_inicio", "fecha_fin", "tipo_temporada", "estatus"]
    for campo in campos_requeridos:
        if campo not in data:
            return jsonify({"error": f"Campo faltante: {campo}"}), 400

    try:
        # Agregar id_administrador al JSON que se enviará
        data["id_administrador"] = g.id_administrador

        nuevo_temporada = service.register(data)
        return jsonify(nuevo_temporada.to_dict()), 201

    except Exception as e:
        return jsonify({"error en controller": str(e)}), 500
