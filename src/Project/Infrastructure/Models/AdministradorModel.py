from sqlalchemy import Column, Integer, String, Time, Text, ForeignKey
from src.DataBases.MySQL import Base


class AdministradorModel(Base):
    __tablename__ = "administrador"
    
    id_administrador = Column(Integer, primary_key=True)
    nombre = Column(String)
    correo = Column(String)
    password = Column(String)

    def to_dict(self):
        return {
            "id_administrador": self.id_administrador,
            "nombre": self.nombre,
            "correo": self.correo,
            "password": self.password
        }
