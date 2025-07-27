from src.Project.Domain.Ports.IComentarioRepository import IComentario
from src.Project.Infrastructure.Models.ComentarioModel import ComentarioModel
from src.DataBases.MySQL import SessionLocal


class ComentarioRepository(IComentario):
    def __init__(self):
        self.db = SessionLocal()
        
    def get_all(self):
        with SessionLocal() as db:
            return db.query(ComentarioModel).all()
        
    def get_by_establecimiento(self, id_establecimiento):
        with SessionLocal() as db:
            return db.query(ComentarioModel).filter(
                ComentarioModel.fk_establecimiento == id_establecimiento
            ).all()

    def create(self, comentario):
        with SessionLocal() as db:
            nuevo = ComentarioModel(
                comentario=comentario["comentario"],
                estrellas_calificacion=comentario["estrellas_calificacion"],
                fk_usuario=comentario["id_usuario"],
                fk_establecimiento=comentario["idalta_establecimiento"]
            )
            db.add(nuevo)
            db.commit()
            db.refresh(nuevo)
            return nuevo

    
