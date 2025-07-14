class EstablecimientoService:
    #Se pasan dos argumentos dado que EstblecimientoService no tenía Create y ahora sí
    def __init__(self, getter, creator, deleter):
        self.getter = getter
        self.creator = creator
        self.deleter = deleter
    def listar(self):
        return self.getter.execute_all()

    def obtener(self, id_):
        return self.getter.execute_by_id(id_)

    def create(self, id_, nombre, direccion, ciudad, tipo, horario, precio, imagen, id_administrador):
        return self.creator.create_establecimiento(id_, nombre, direccion, ciudad, tipo, horario, precio, imagen, id_administrador)

    def delete(self, id_):
        return self.deleter.delete(id_)