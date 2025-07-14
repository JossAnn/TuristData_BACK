from abc import ABC, abstractmethod


class IEventosEspecialesRepository(ABC):
    @abstractmethod
    def create(self, eventos_especiales):
        pass
