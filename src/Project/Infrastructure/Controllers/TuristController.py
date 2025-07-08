from flask import Blueprint, jsonify, request
from src.Project.Infrastructure.Repositories.TuristRepository import (
    TuristRepository
)
from src.Project.Aplication.TuristUseCases.GetTurist import GetTurist
from src.Project.Infrastructure.Services.TuristService import (TuristService)


bp_turista = Blueprint("turista", __name__)
service= TuristService(GetTurist(TuristRepository()))

@bp_turista.route("/turistas/<int:id_>", methods=["GET"])
def obtener_turista(id_):
    print("Obteniendo turista controller por ID:", id_)
    
    est = service.obtener(id_)
    print("Obteniendo :", est.to_dict())
    return jsonify(est.to_dict()) if est else ("Not Found", 404)
