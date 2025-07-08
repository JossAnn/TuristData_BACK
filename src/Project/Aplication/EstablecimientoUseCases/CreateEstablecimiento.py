class CreateEstablecimiento:
    def __init__(self, repository):
        self.establecimiento_repository = repository

    def create_establecimiento(self, establecimiento):
        return self.establecimiento_repository.create_establecimiento(establecimiento)
