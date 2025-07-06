from flask import Blueprint, jsonify, request

from src.Project.Infrastructure.Repositories.AdministradoRepository import(
    AdministradorRepository
)
from src.Project.Aplication.AdministradorUseCases.CreateAdministrador import CreateAdministrador
from src.Project.Aplication.AdministradorUseCases.LoginAdministrador import LoginAdministrador
from src.Project.Infrastructure.Services.AdministradorService import (
    AdministradorService,
)


bp_administrador = Blueprint("administrador", __name__)
service = AdministradorService(CreateAdministrador(AdministradorRepository()),LoginAdministrador(AdministradorRepository()))


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

@bp_administrador.route("/administrador/login", methods=["POST"])
def login_administrador():
    data = request.get_json()
    if not data or "correo" not in data or "password" not in data:
        return jsonify({"error": "Correo y contraseña requeridos"}), 400

    try:
        admin = service.login(data["correo"], data["password"])
        print("Login exitoso - ControllerLoginAdmin:", admin)

        return jsonify({
            "message": "Inicio de sesión exitoso",
            "correo": admin.correo,
            "password": admin.password  # ⚠️ Solo para pruebas. NUNCA en producción.
        }), 200

    except ValueError as e:
        return jsonify({"error": str(e)}), 401
    except Exception as e:
        print("Error en login - ControllerLoginAdmin:", str(e))
        return jsonify({"error": "Error interno del servidor"}), 500
