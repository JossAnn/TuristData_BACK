class CreateEstablecimiento:
    def __init__(self, repository):
        self.establecimiento_repository = repository

    def register_establecimiento(self, establecimiento):
        return self.establecimiento_repository.resgister_establecimiento(establecimiento)
