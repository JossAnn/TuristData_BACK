# class CreateEstablecimiento:
#     def __init__(self, repository):
#         self.establecimiento_repository = repository

#     def create_establecimiento(self, establecimiento):
#         return self.establecimiento_repository.create_establecimiento(establecimiento)

class CreateEstablecimiento:
    def __init__(self, repository):
        self.repository = repository

    def create_establecimiento(self, nombre, direccion, ciudad, estado, tipo, horario, precio, imagen, id_administrador):
        return self.repository.create(
            nombre, direccion, ciudad, estado, tipo, horario, precio, imagen, id_administrador
        )