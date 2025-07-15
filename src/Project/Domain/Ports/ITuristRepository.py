from abc import ABC, abstractmethod


class IturistRepository(ABC):
    @abstractmethod
    def get_user_by_id(self, id_):
        pass

    # @abstractmethod
    # def create(self, turista):
    #     pass
    
    @abstractmethod
    def create(self, nombre, correo, password):
        pass

    @abstractmethod
    def login_turist_correo_password(self, correo, password):
        pass
