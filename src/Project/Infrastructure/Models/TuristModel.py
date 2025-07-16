from sqlalchemy import Column, Integer, String, Time, Text, ForeignKey
from src.DataBases.MySQL import Base


class TuristModel(Base):
    __tablename__ = "turista"
    
    id_usuario = Column(Integer, primary_key=True)
    nombre = Column(String)
    correo = Column(String, nullable=True, unique=True)
    password = Column(String)

    def to_dict(self):
        return {
            "id_usuario": self.id_usuario,
            "nombre": self.nombre,
            "correo": self.correo,
            "password": self.password
        }
