class GetEventosEspecialesUseCase:
    def __init__(self, repository):
        self.repository = repository

    def execute_all(self):
        return self.repository.get_all()

    # def execute_by_id(self, id_):
    #     return self.repository.get_by_id(id_)
