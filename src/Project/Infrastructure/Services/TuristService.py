class TuristService:
    def __init__(self, use_case):
        self.use_case = use_case

    def obtener(self, id_):
        #print("Obteniendo turista por ID:", id_)
        return self.use_case.get_user_by_id(id_)
