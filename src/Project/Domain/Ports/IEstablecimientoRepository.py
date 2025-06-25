from abc import ABC, abstractmethod


class IEstablecimientoRepository(ABC):
    @abstractmethod
    def get_all(self):
        pass

    @abstractmethod
    def get_by_id(self, id_):
        pass

    @abstractmethod
    def create(self, establecimiento):
        pass
