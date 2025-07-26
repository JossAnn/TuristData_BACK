from sqlalchemy import Column, Integer, String, Time, Text, ForeignKey
from src.DataBases.MySQL import Base


class ComentarioModel(Base):
    __tablename__ = "comentarios"
    
    id_comentarios = Column(Integer, primary_key=True, autoincrement=True)
    comentario = Column(String)
    fk_usuario = Column(Integer, ForeignKey("turista.id_usuario"))
    fk_establecimiento = Column(Integer, ForeignKey("alta_establecimiento.idalta_establecimiento"))
    fk_administrador = Column(Integer, ForeignKey("administrador.id_administrador"))

    def to_dict(self):
        return {
            "id_comentarios": self.id_comentarios,
            "comentario": self.comentario
        }
