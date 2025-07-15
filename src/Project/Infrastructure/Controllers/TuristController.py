from flask import Blueprint, jsonify, request
from src.Project.Infrastructure.Repositories.TuristRepository import (
    TuristRepository
)

from src.Project.Aplication.TuristUseCases.CreateTurist import CreateTurist
from src.Project.Aplication.TuristUseCases.GetTurist import GetTurist
from src.Project.Aplication.TuristUseCases.LoginTurista import LognTurista
from src.Project.Infrastructure.Services.TuristService import (TuristService)

from src.Project.Infrastructure.Utils.jwt_utils import crear_token


bp_turista = Blueprint("turistas", __name__)
service= TuristService(GetTurist(TuristRepository()), CreateTurist(TuristRepository()), LognTurista(TuristRepository()))


@bp_turista.route("/turistas/<int:id_>", methods=["GET"])
def obtener_turista(id_):
    print("Obteniendo turista controller por ID:", id_)
    
    est = service.obtener(id_)
    print("Obteniendo :", est.to_dict())
    return jsonify(est.to_dict()) if est else ("Not Found", 404)

@bp_turista.route("/turistas", methods=["POST"])
def registrar_turista():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    try:
        nuevo_turista = service.register(data)  # ✅ corregido aquí
        # return jsonify(nuevo_turista.__dict__), 201
        return jsonify(nuevo_turista.to_dict()), 201
    except Exception as e:
        print("Error al registrar turista - Controller:", str(e))
        return jsonify({"error": str(e)}), 500
    
@bp_turista.route("/turistas/login", methods=["POST"])
def login_turista():
    data = request.get_json()
    if not data or "correo" not in data or "password" not in data:
        return jsonify({"error": "Correo y contraseña requeridos"}), 400

    try:
        turist = service.login(data["correo"], data["password"])

        token = crear_token(turist.id_usuario, turist.correo)

        return jsonify({
            "message": "Inicio de sesión exitoso",
            "token": token
        }), 200

    except ValueError as e:
        return jsonify({"error": str(e)}), 401
    except Exception as e:
        print("Error en login - ControllerLoginAdmin:", str(e))
        return jsonify({"error": "Error interno del servidor"}), 500