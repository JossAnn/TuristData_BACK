from src.Project.Domain.Ports.ILugares import ILugares 
from src.Project.Infrastructure.Models.lugar_turistico import LugarTuristico
from src.DataBases.MySQL import SessionLocal


class LugaresRepository(ILugares):
    def __init__(self):
        self.db = SessionLocal()
        
    def create(self, luagres):
        nuevo = LugarTuristico(
            nombre=luagres["nombre"],
            estado=luagres["estado"]
        )
        self.db.add(nuevo)
        self.db.commit()
        self.db.refresh(nuevo)
        return nuevo
    
