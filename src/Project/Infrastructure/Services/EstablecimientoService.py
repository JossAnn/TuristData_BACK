class EstablecimientoService:
    def __init__(self, use_case):
        self.use_case = use_case

    def listar(self):
        return self.use_case.execute_all()

    def obtener(self, id_):
        return self.use_case.execute_by_id(id_)
