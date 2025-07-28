from sqlalchemy import Column, Integer, String
from src.DataBases.MySQL import Base

class LugarTuristico(Base):
    __tablename__ = "lugares"

    id_lugares = Column(Integer, primary_key=True, index=True)
    nombre = Column(String(255), nullable=False)
    estado = Column(String(100), nullable=False)


    def to_dict(self):
        return {
            "nombre": self.nombre,
            "estado": self.estado
        }