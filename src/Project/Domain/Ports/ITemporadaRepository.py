from abc import ABC, abstractmethod


class ITemporadaRepository(ABC):
    @abstractmethod
    def create(self, temporada):
        pass
