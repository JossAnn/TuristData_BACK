from src.Project.Domain.Ports.IEstablecimientoRepository import IEstablecimientoRepository
from src.Project.Infrastructure.Models.EstablecimientoModel import EstablecimientoModel
from src.DataBases.MySQL import SessionLocal


class EstablecimientoRepository(IEstablecimientoRepository):
    # def __init__(self):
    #     self.db = SessionLocal()

    def get_all(self):
        with SessionLocal() as db:
            return self.db.query(EstablecimientoModel).all()

    def get_by_id(self, id_):
        with SessionLocal() as db:
            return (
                self.db.query(EstablecimientoModel)
                .filter_by(idalta_establecimiento=id_)
                .first()
            )
    
    def create(self, nombre, direccion, ciudad, tipo, horario, precio, imagen, id_administrador):
        with SessionLocal() as db:
            nuevo = EstablecimientoModel(
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
    
    def put(self, id_, nombre, direccion, ciudad, tipo, horario, precio, imagen):
        obj = self.db.query(EstablecimientoModel).filter_by(idalta_establecimiento=id_).first()
        if obj:
            obj.nombre = nombre
            obj.direccion = direccion
            obj.ciudad = ciudad
            obj.tipo = tipo
            obj.horario = horario
            obj.precio = precio
            obj.imagen = imagen
            self.db.commit()
            self.db.refresh(obj)
            return obj
        return None
