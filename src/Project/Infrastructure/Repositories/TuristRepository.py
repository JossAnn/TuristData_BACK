from src.Project.Domain.Ports.ITuristRepository import IturistRepository
from src.Project.Infrastructure.Models.TuristModel import TuristModel
from src.DataBases.MySQL import SessionLocal


class TuristRepository(IturistRepository):
    def __init__(self):
        self.db = SessionLocal()
    def get_user_by_id(self, id_):
        return (
            self.db.query(TuristModel)
            .filter_by(id_usuario=id_)
            .first()
        )

    def create(self, turista):
        nuevo = TuristModel(            
            nombre=turista["nombre"],
            correo=turista["correo"],
            password=turista["password"]
        )
        self.db.add(nuevo)
        self.db.commit()
        self.db.refresh(nuevo)
        return nuevo
    
    def login_turist_correo_password(self, correo, password):
        # Buscar al turista por correo
        turist = self.db.query(TuristModel).filter_by(correo=correo).first()
        return turist 
    
