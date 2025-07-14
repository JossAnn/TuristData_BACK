class PutEstablecimiento:
    def __init__(self, repository):
        self.repository = repository

    def Actualizar(self, id_, nombre, direccion, ciudad, tipo, horario, precio, imagen):
        return self.repository.put(
            id_, nombre, direccion, ciudad, tipo, horario, precio, imagen
        )
        
        