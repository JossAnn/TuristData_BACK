from src.Project.Infrastructure.Services.EncriptService import EncriptServices

class CreateTurist:
    def __init__(self, repository):
        self.turist_repository = repository

    # def create_turist(self, user):
    #     return self.turist_repository.create_user(user)

    def registrar_turist(self, data):
        data["password"] = EncriptServices.encode_password(data["password"])
        return self.turist_repository.create(data)