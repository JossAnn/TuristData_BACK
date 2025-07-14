from sqlalchemy import Column, Integer, String, Time, Text, ForeignKey
from src.DataBases.MySQL import Base


class TemporadaModel(Base):
    __tablename__ = "temporadas"
    
    id_temporadas = Column(Integer, primary_key=True)
    nombre = Column(String)
    fecha_inicio = Column(String)
    fecha_fin = Column(String)
    tipo_temporada = Column(Integer)
    estatus = Column(Integer)

    def to_dict(self):
        return {
            "id_temporadas": self.id_temporadas,
            "nombre": self.nombre,
            "fecha_inicio": self.fecha_inicio,
            "fecha_fin": self.fecha_fin,
            "tipo_temporada": self.tipo_temporada,
            "estatus": self.estatus
        }
