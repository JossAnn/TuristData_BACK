from src.Project.Domain.Ports.IComentarioRepository import IComentario
from src.Project.Infrastructure.Models.ComentarioModel import ComentarioModel
from src.DataBases.MySQL import SessionLocal

from sqlalchemy.orm import joinedload
from src.Project.Infrastructure.Models.TuristModel import TuristModel 

class ComentarioRepository(IComentario):
    def __init__(self):
        self.db = SessionLocal()
        
    def get_all(self):
        with SessionLocal() as db:
            return db.query(ComentarioModel).all()
        
    def get_by_establecimiento(self, id_establecimiento):
        with SessionLocal() as db:
            comentarios = (
                db.query(ComentarioModel)
                .options(joinedload(ComentarioModel.usuario))
                .filter(ComentarioModel.fk_establecimiento == id_establecimiento)
                .all()
            )

            resultado = []
            for comentario in comentarios:
                c_dict = {
                    "id_comentarios": comentario.id_comentarios,
                    "estrellas_calificacion": comentario.estrellas_calificacion,
                    "comentario": comentario.comentario,
                    "nombre_usuario": comentario.usuario.nombre if comentario.usuario else None
                }
                resultado.append(c_dict)
            return resultado

        
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

            usuario = db.query(TuristModel).filter(TuristModel.id_usuario == nuevo.fk_usuario).first()

            return {
                "id_comentarios": nuevo.id_comentarios,
                "comentario": nuevo.comentario,
                "estrellas_calificacion": nuevo.estrellas_calificacion,
                "fk_usuario": nuevo.fk_usuario,
                "fk_establecimiento": nuevo.fk_establecimiento,
                "nombre_usuario": usuario.nombre if usuario else None,
            }

    
