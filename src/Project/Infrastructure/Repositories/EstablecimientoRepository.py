from src.Project.Domain.Ports.IEstablecimientoRepository import IEstablecimientoRepository
from src.Project.Infrastructure.Models.EstablecimientoModel import EstablecimientoModel
from src.DataBases.MySQL import SessionLocal


class EstablecimientoRepository(IEstablecimientoRepository):
    def __init__(self):
        self.db = SessionLocal()

    def get_all(self):
        return self.db.query(EstablecimientoModel).all()

    def get_by_id(self, id_):
        return (
            self.db.query(EstablecimientoModel)
            .filter_by(idalta_establecimiento=id_)
            .first()
        )

    def create(self, data):
        nuevo = EstablecimientoModel(**data)
        self.db.add(nuevo)
        self.db.commit()
        self.db.refresh(nuevo)
        return nuevo
