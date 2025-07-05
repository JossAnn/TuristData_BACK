from flask import Blueprint, jsonify, request

from src.Project.Infrastructure.Repositories.AdministradoRepository import(
    AdministradorRepository
)
from src.Project.Aplication.AdministradorUseCases.CreateAdministrador import CreateAdministrador
from src.Project.Infrastructure.Services.AdministradorService import (
    AdministradorService,
)


bp_administrador = Blueprint("administrador", __name__)
service = AdministradorService(CreateAdministrador(AdministradorRepository()))


# @bp_administrador.route("/administrador", methods=["POST"])
# def registrar_administrador():
#     data = request.get_json()
#     if not data:
#         return jsonify({"error": "No data provided"}), 400

#     try:
#         nuervo_administrador = service.register(
#             data["id_administrador"],
#             data["nombre"],
#             data["correo"],
#             data["password"]
#         )
#         return jsonify(nuervo_administrador.__dict__), 201
#     except Exception as e:
#         print("Error al registrar administrador - Controller:", str(e))
#         return jsonify({"error": str(e)}), 500

@bp_administrador.route("/administrador", methods=["POST"])
def registrar_administrador():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    try:
        nuevo_administrador = service.register(data)  # ✅ corregido aquí
        # return jsonify(nuevo_administrador.__dict__), 201
        return jsonify(nuevo_administrador.to_dict()), 201
    except Exception as e:
        print("Error al registrar administrador - Controller:", str(e))
        return jsonify({"error": str(e)}), 500

