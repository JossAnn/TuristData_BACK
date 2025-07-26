from abc import ABC, abstractmethod


class ILugares(ABC):
    @abstractmethod
    def create(self, lugares):
        pass
