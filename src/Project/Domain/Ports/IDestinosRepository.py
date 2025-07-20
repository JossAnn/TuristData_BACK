from abc import ABC, abstractmethod


class IDestinosRepository(ABC):
    #Mostrar todos los establecimientos
    @abstractmethod
    def get_all(self):
        pass
    @abstractmethod
    def create(self, destino):
        pass
    
