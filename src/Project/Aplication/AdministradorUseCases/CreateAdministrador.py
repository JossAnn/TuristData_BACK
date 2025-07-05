from src.Project.Infrastructure.Services.EncriptService import EncriptServices

class CreateAdministrador:
    def __init__(self, repository):
        self.administrador_repository = repository

    # def registrar_administrador(self, administrador):
    #     return self.administrador_repository.registrar_administrador(administrador)
    def registrar_administrador(self, data):
        data["password"] = EncriptServices.encode_password(data["password"])
        return self.administrador_repository.create(data)