from abc import ABC, abstractmethod


class IAdminRepository(ABC):

    @abstractmethod
    def create(self, nombre, correo, password):
        pass
    
    @abstractmethod
    def login_correo_password(self, correo, password):
        pass