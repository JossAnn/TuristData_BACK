from src.Project.Domain.Ports.IEventosEspecialesRepository import IEventosEspecialesRepository
from src.Project.Infrastructure.Models.EventosEspecialesModel import EventosEspecialesModel
from src.DataBases.MySQL import SessionLocal


class EventosEspecialesRepository(IEventosEspecialesRepository):
    def __init__(self):
        self.db = SessionLocal()
        
    def get_all(self):
        with SessionLocal() as db:
            return db.query(EventosEspecialesModel).all()

    def create(self, eventos_especiales):
        nuevo = EventosEspecialesModel(
            nombre=eventos_especiales["nombre"],
            fecha_inicio=eventos_especiales["fecha_inicio"],
            fecha_final=eventos_especiales["fecha_final"],
            descripcion=eventos_especiales["descripcion"],
            estado_afectado=eventos_especiales["estado_afectado"],
            id_destino=eventos_especiales["id_destino"],
            id_temporada=eventos_especiales["id_temporada"],
            id_administrador=eventos_especiales["id_administrador"]

        )
        self.db.add(nuevo)
        self.db.commit()
        self.db.refresh(nuevo)
        return nuevo
    
