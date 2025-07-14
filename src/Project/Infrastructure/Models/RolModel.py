from sqlalchemy import Column, Integer, String
from src.DataBases.MySQL import Base


class RolModel(Base):
    __tablename__ = "rol"
    
    id_rol = Column(Integer, primary_key=True)
    nombre_rol = Column(String)

    def to_dict(self):
        return {
            "id_rol": self.id_rol,
            "nombre_rol": self.nombre_rol
        }
