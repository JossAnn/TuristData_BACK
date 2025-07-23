from sqlalchemy import Column, Integer, String, Time, Text, ForeignKey
from src.DataBases.MySQL import Base


class EstablecimientoModel(Base):
    __tablename__ = "alta_establecimiento"
    __table_args__ = {'extend_existing': True}

    idalta_establecimiento = Column(Integer, primary_key=True, autoincrement=True)
    nombre = Column(String(45), nullable=False)
    direccion = Column(String(45), nullable=False)
    ciudad = Column(String(45), nullable=False)
    tipo = Column(String(45))
    horario = Column(String(45))
    precio = Column(String(45))
    imagen = Column(Text)
    id_administrador = Column(Integer, ForeignKey("administrador.id_administrador"))
    
    def to_dict(self):
        return {
            "idalta_establecimiento": self.idalta_establecimiento,
            "nombre": self.nombre,
            "direccion": self.direccion,
            "ciudad": self.ciudad,
            "tipo": self.tipo,
            "horario":self.horario,
            "precio": self.precio,
            "imagen": self.imagen,
            # "id_administrador": self.id_administrador
        }

