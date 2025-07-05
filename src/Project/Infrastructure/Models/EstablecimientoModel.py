from sqlalchemy import Column, Integer, String, Time, Text, ForeignKey
from src.DataBases.MySQL import Base


class EstablecimientoModel(Base):
    __tablename__ = "alta_establecimiento"
    __table_args__ = {'extend_existing': True}

    idalta_establecimiento = Column(Integer, primary_key=True, autoincrement=True)
    nombre = Column(String(45), nullable=False)
    direccion = Column(String(45), nullable=False)
    ciudad = Column(String(45), nullable=False)
    id_tipo = Column(Integer, ForeignKey("Tipo.id_tipo"))
    horario = Column(Time)
    precio = Column(String(45))
    imagen = Column(Text)
    id_administrador = Column(Integer, ForeignKey("administrador.id_administrador"))
    
    def to_dict(self):
        return {
            "idalta_establecimiento": self.idalta_establecimiento,
            "nombre": self.nombre,
            "direccion": self.direccion,
            "ciudad": self.ciudad,
            "id_tipo": self.id_tipo,
            "horario": str(self.horario) if self.horario else None,
            "precio": self.precio,
            "imagen": self.imagen,
            "id_administrador": self.id_administrador
        }

    """import uuid
from sqlalchemy.dialects.mysql import CHAR
from sqlalchemy import Column, String, Time, Text, ForeignKey
from sqlalchemy.types import TypeDecorator
from src.DataBases.MySQL import Base


class GUID(TypeDecorator):
    impl = CHAR(36)

    def process_bind_param(self, value, dialect):
        if value is None:
            return str(uuid.uuid4())
        return str(value)

    def process_result_value(self, value, dialect):
        return str(value)


class EstablecimientoModel(Base):
    __tablename__ = "alta_establecimiento"

    idalta_establecimiento = Column(GUID(), primary_key=True, default=uuid.uuid4)
    nombre = Column(String(45), nullable=False)
    direccion = Column(String(45), nullable=False)
    ciudad = Column(String(45), nullable=False)
    id_tipo = Column(
        String(36), ForeignKey("Tipo.id_tipo")
    )  # También debe ser UUID si tipo cambia
    horario = Column(Time)
    precio = Column(String(45))
    imagen = Column(Text)

    """
