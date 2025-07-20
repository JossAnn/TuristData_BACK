from sqlalchemy import Column, Integer, String, Time, Text, ForeignKey
from src.DataBases.MySQL import Base


class DestinosModel(Base):
    __tablename__ = "destinos"
    
    id_destinos = Column(Integer, primary_key=True, autoincrement=True)
    nombre = Column(String)
    estado = Column(String)
    id_turista = Column(Integer, ForeignKey("turista.id_usuario"))

    def to_dict(self):
        return {
            "id_destinos": self.id_destinos,
            "nombre": self.nombre,
            "estado": self.estado,
        }
