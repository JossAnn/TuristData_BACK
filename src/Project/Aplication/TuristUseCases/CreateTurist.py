class CreateTurist:
    def __init__(self, repository):
        self.turist_repository = repository

    def create_turist(self, user):
        return self.turist_repository.create_user(user)
