from src.Project.Infrastructure.Services.EncriptService import EncriptServices

class LognTurista:
    def __init__(self, repository):
        self.turist_repository = repository
    def ejecutar(self, correo, password):
        turista = self.turist_repository.login_turist_correo_password(correo, password)

        if not turista:
            raise ValueError("Correo no encontrado")

        if not EncriptServices.auth_password(password, turista.password):
            raise ValueError("Contraseña incorrecta")

        return turista  