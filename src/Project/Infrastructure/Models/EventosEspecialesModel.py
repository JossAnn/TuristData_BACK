from sqlalchemy import Column, Integer, String
from src.DataBases.MySQL import Base


class EventosEspecialesModel(Base):
    __tablename__ = "eventos_especiales"
    
    idEventos_especiales = Column(Integer, primary_key=True)
    nombre = Column(String)
    fecha_inicio = Column(DateTime)
    fecha_final = Column(DateTime)
    descripcion = Column(String)
    estado_afectado = Column(String)
    id_destino = Column(Integer, ForeignKey("destinos.id_destinos"))
    id_temporada = Column(Integer, ForeignKey("temporadas.id_temporadas"))
    id_administrador = Column(Integer, ForeignKey("temporadas.id_administrador"))
    def to_dict(self):
        return {
            "idEventos_especiales": self.idEventos_especiales,
            "nombre": self.nombre,
            "fecha_inicio": self.fecha_inicio,
            "fecha_final": self.fecha_final,
            "descripcion": self.descripcion,
            "estado_afectado": self.estado_afectado
        }
