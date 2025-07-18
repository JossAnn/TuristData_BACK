from src.Project.Domain.Ports.ITuristRepository import IturistRepository
from src.Project.Infrastructure.Models.TuristModel import TuristModel
from src.DataBases.MySQL import SessionLocal

class TuristRepository(IturistRepository):
    def get_user_by_id(self, id_):
        with SessionLocal() as db:
            return db.query(TuristModel).filter_by(id_usuario=id_).first()

    def create(self, turista):
        with SessionLocal() as db:
            nuevo = TuristModel(            
                nombre=turista["nombre"],
                correo=turista["correo"],
                password=turista["password"]
            )
            db.add(nuevo)
            db.commit()
            db.refresh(nuevo)
            return nuevo
    
    def login_turist_correo_password(self, correo, password):
        with SessionLocal() as db:
            turist = db.query(TuristModel).filter_by(correo=correo).first()
            return turist