from abc import ABC, abstractmethod

class IComentario(ABC):
    @abstractmethod
    def get_all(self):
        pass
    
    @abstractmethod
    def get_by_establecimiento(self, id_establecimiento):
        pass

    @abstractmethod
    def create(self, comentario):
        pass

