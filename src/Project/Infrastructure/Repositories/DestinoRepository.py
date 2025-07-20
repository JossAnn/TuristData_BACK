from src.Project.Domain.Ports.IDestinosRepository import IDestinosRepository
from src.Project.Infrastructure.Models.DestinosModel import DestinosModel
from src.DataBases.MySQL import SessionLocal


class DestinosRepository(IDestinosRepository):
    def __init__(self):
        self.db = SessionLocal()
        
    def get_all(self):
        with SessionLocal() as db:
            return db.query(DestinosModel).all()

    def create(self, destinos):
        nuevo = DestinosModel(
            nombre=destinos["nombre"],
            estado=destinos["estado"],
            id_turista=destinos["id_turista"]
        )
        self.db.add(nuevo)
        self.db.commit()
        self.db.refresh(nuevo)
        return nuevo
    
