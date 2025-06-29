from src.Project.Domain.Ports.ITuristRepository import IturistRepository
from src.Project.Infrastructure.Models.TuristModel import TuristModel
from src.DataBases.MySQL import SessionLocal


class TuristRepository(IturistRepository):
    def __init__(self):
        self.db = SessionLocal()
    def get_by_id(self, id_):
        return (
            self.db.query(TuristModel)
            .filter_by(idalta_establecimiento=id_)
            .first()
        )

    def create(self, data):
        nuevo = TuristModel(**data)
        self.db.add(nuevo)
        self.db.commit()
        self.db.refresh(nuevo)
        return nuevo
