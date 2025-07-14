from abc import ABC, abstractmethod


class IEstablecimientoRepository(ABC):
    #Mostrar todos los establecimientos
    @abstractmethod
    def get_all(self):
        pass
    
    #Mostrar un establecimiento por id
    @abstractmethod
    def get_by_id(self, id_):
        pass

    #Agregar establecimientos por administrador
    @abstractmethod
    def create(self, establecimiento):
        pass
    
    #Eliminar un establecimiento por id
    @abstractmethod
    def delete(self, id_):
        pass
    
    #Actualizar un establecimiento por id
    @abstractmethod
    def put(self, id_, nombre, direccion, ciudad, tipo, horario, precio, imagen):
        pass

    
