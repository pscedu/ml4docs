from django.db import models


class Images(models.Model):
    name = models.CharField(max_length=200)
    verified = models.BooleanField(default=False)
    human_tagged = models.BooleanField(default=False)
    model_tagged = models.BooleanField(default=False)
    pending = models.BooleanField(default=True)

    def set_verified(self):
        self.verified, self.pending = True, False

    def set_model_tagged(self):
        self.model_tagged, self.human_tagged, self.pending = True, False, True

    def set_human_tagged(self):
        self.model_tagged, self.human_tagged = False, True
        self.set_verified()
