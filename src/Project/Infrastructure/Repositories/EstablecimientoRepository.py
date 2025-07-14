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
    
    def create(self, id_, nombre, direccion, ciudad, tipo, horario, precio, imagen, id_administrador):
        nuevo = EstablecimientoModel(
            idalta_establecimiento=id_,
            nombre=nombre,
            direccion=direccion,
            ciudad=ciudad,
            tipo=tipo,
            horario=horario,
            precio=precio,
            imagen=imagen,
            id_administrador=id_administrador
        )
        self.db.add(nuevo)
        self.db.commit()
        self.db.refresh(nuevo)
        return nuevo

    def delete(self, id_):
        obj = self.db.query(EstablecimientoModel).filter_by(idalta_establecimiento=id_).first()
        if obj:
            self.db.delete(obj)
            self.db.commit()
            return obj  # opcionalmente puedes devolver el objeto eliminado
        return None
