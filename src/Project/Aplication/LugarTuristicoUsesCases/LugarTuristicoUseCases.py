from src.Project.Infrastructure.Services.GeoapifyService import GeoapifyService

class LugarTuristicoUseCases:
    def __init__(self):
        self.service = GeoapifyService()

    def ejecutar(self):
        self.service.obtener_y_guardar_lugares()
