class EstablecimientoService:
    #Se pasan dos argumentos dado que EstblecimientoService no tenía Create y ahora sí
    def __init__(self, getter, creator, deleter, updater):
        self.getter = getter
        self.creator = creator
        self.deleter = deleter
        self.updater = updater
    def listar(self):
        return self.getter.execute_all()

    def obtener(self, id_):
        return self.getter.execute_by_id(id_)

    def create(self, nombre, direccion, ciudad, estado, tipo, horario, precio, imagen, id_administrador):
        return self.creator.create_establecimiento(nombre, direccion, ciudad, estado, tipo, horario, precio, imagen, id_administrador)

    def delete(self, id_):
        return self.deleter.delete(id_)
    
    def put(self, id_, nombre, direccion, ciudad, tipo, horario, precio, imagen):
        return self.updater.Actualizar(id_, nombre, direccion, ciudad, tipo, horario, precio, imagen)
    def listar_por_administrador(self, id_administrador):
        return self.getter.obtener_por_administrador(id_administrador)

    def listar_por_estado(self, estado):
        return self.getter.obtener_por_estado(estado)
