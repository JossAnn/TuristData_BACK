from src.Project.Domain.Ports.lAdminRepository import IAdminRepository
from src.Project.Infrastructure.Models.AdministradorModel import AdministradorModel
from src.DataBases.MySQL import SessionLocal


class AdministradorRepository(IAdminRepository):
    def __init__(self):
        self.db = SessionLocal()

    def create(self, administrador):
        nuevo = AdministradorModel(
            nombre=administrador["nombre"],
            correo=administrador["correo"],
            password=administrador["password"]
        )
        self.db.add(nuevo)
        self.db.commit()
        self.db.refresh(nuevo)
        return nuevo
