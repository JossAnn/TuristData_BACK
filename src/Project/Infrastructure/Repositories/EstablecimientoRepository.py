from src.Project.Domain.Ports.IEstablecimientoRepository import IEstablecimientoRepository
from src.Project.Infrastructure.Models.EstablecimientoModel import EstablecimientoModel
from src.DataBases.MySQL import SessionLocal


class EstablecimientoRepository(IEstablecimientoRepository):

    def get_all(self):
        with SessionLocal() as db:
            return db.query(EstablecimientoModel).all()

    def get_by_id(self, id_):
        with SessionLocal() as db:
            return db.query(EstablecimientoModel).filter_by(idalta_establecimiento=id_).first()
    
    def create(self, nombre, direccion, ciudad, estado, tipo, horario, precio, imagen, id_administrador):
        with SessionLocal() as db:
            nuevo = EstablecimientoModel(
                nombre=nombre,
                direccion=direccion,
                ciudad=ciudad,
                estado=estado,
                tipo=tipo,
                horario=horario,
                precio=precio,
                imagen=imagen,
                id_administrador=id_administrador
            )
            db.add(nuevo)
            db.commit()
            db.refresh(nuevo)
            return nuevo

    def delete(self, id_):
        with SessionLocal() as db:
            obj = db.query(EstablecimientoModel).filter_by(idalta_establecimiento=id_).first()
            if obj:
                db.delete(obj)
                db.commit()
                return obj
            return None
    
    def put(self, id_, nombre, direccion, ciudad, tipo, horario, precio, imagen):
        with SessionLocal() as db:
            obj = db.query(EstablecimientoModel).filter_by(idalta_establecimiento=id_).first()
            if obj:
                obj.nombre = nombre
                obj.direccion = direccion
                obj.ciudad = ciudad
                obj.tipo = tipo
                obj.horario = horario
                obj.precio = precio
                obj.imagen = imagen
                db.commit()
                db.refresh(obj)
                return obj
            return None

    def get_by_administrador(self, id_administrador):
        with SessionLocal() as db:
            return db.query(EstablecimientoModel).filter_by(id_administrador=id_administrador).all()

    def get_by_estado(self, estado):
        with SessionLocal() as db:
            return db.query(EstablecimientoModel).filter_by(estado=estado).all()
