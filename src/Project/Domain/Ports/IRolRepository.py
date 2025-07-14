from abc import ABC, abstractmethod


class IRolRepository(ABC):
    # @abstractmethod
    # def get_all(self):
    #     pass

    # @abstractmethod
    # def get_by_id(self, id_):
    #     pass

    @abstractmethod
    def create(self, rol):
        pass
    
    # @abstractmethod
    # def register(self, establecimiento):
    #     pass
