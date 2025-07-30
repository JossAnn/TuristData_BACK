from sqlalchemy import Column, Integer, String, Time, Text, ForeignKey, Date
from src.DataBases.MySQL import Base


class TemporadaModel(Base):
    __tablename__ = "temporadas"
    
    id_temporadas = Column(Integer, primary_key=True, autoincrement=True)
    nombre = Column(String)
    fecha_inicio = Column(Date)
    fecha_fin = Column(Date)
    tipo_temporada = Column(String)
    estatus = Column(String)

    def to_dict(self):
        return {
            "nombre": self.nombre,
            "fecha_inicio": self.fecha_inicio,
            "fecha_fin": self.fecha_fin,
            "tipo_temporada": self.tipo_temporada,
            "estatus": self.estatus
        }
