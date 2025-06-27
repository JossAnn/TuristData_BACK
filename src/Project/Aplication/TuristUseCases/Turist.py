class TuristUsesCases:
    def __init__(self, repository):
        self.turit_repository = repository

    def create_user(self, user):
        return self.turit_repository.create_user(user)

    # def get_user_by_id(self, user_id):
    #     return self.user_repository.get_user_by_id(user_id)

    def update_user(self, user):
        return self.turit_repository.update_user(user)

    def delete_user(self, user_id):
        return self.turit_repository.delete_user(user_id)

    # def get_all_users(self):
    #     return self.user_repository.get_all_users()