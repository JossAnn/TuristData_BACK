from flask import Blueprint, jsonify, request
from src.Project.Infrastructure.Repositories.TuristRepository import (
    TuristRepository
)

from src.Project.Aplication.TuristUseCases.CreateTurist import CreateTurist
from src.Project.Aplication.TuristUseCases.GetTurist import GetTurist
from src.Project.Infrastructure.Services.TuristService import (TuristService)


bp_turista = Blueprint("turistas", __name__)
service= TuristService(GetTurist(TuristRepository()), CreateTurist(TuristRepository()))


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
        nuevo_administrador = service.register(data)  # ✅ corregido aquí
        # return jsonify(nuevo_turista.__dict__), 201
        return jsonify(nuevo_administrador.to_dict()), 201
    except Exception as e:
        print("Error al registrar turista - Controller:", str(e))
        return jsonify({"error": str(e)}), 500