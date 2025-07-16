from src.Project.Domain.Ports.ITemporadaRepository import ITemporadaRepository
from src.Project.Infrastructure.Models.TemporadaModel import TemporadaModel
from src.DataBases.MySQL import SessionLocal

class TemporadaRepository(ITemporadaRepository):
    def __init__(self):
        self.db = SessionLocal()

    def create(self, temporada_data: dict):
        nuevo = TemporadaModel(
            nombre=temporada_data["nombre"],
            fecha_inicio=temporada_data["fecha_inicio"],
            fecha_fin=temporada_data["fecha_fin"],
            tipo_temporada=temporada_data["tipo_temporada"],
            estatus=temporada_data["estatus"]
        )
        self.db.add(nuevo)
        self.db.commit()
        self.db.refresh(nuevo)
        return nuevo
