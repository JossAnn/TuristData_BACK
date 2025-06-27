from sqlalchemy import Column, Integer, String, Time, Text, ForeignKey
from src.DataBases.MySQL import Base


class TuristModel(Base):
    __tablename__ = "alta_establecimiento"

    idusuario = Column(Integer, primary_key=True, autoincrement=True)
    nombre = Column(String(45), nullable=False)
    correo = Column(String(45), nullable=False)
    password = Column(String(45), nullable=False)
    id_rol = Column(Integer, ForeignKey("Rol.id_rol"))

