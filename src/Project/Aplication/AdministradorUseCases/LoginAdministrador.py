from src.Project.Infrastructure.Services.EncriptService import EncriptServices

class LoginAdministrador:
    def __init__(self, repository):
        self.administrador_repository = repository
    def ejecutar(self, correo, password):
        admin = self.administrador_repository.login_correo_password(correo, password)

        if not admin:
            raise ValueError("Correo no encontrado")

        if not EncriptServices.auth_password(password, admin.password):
            raise ValueError("Contraseña incorrecta")

        return admin