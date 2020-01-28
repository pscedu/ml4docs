from django.http import HttpResponse, JsonResponse
from django.template import loader
from django.views import View

import labelMeNav.constants as constants
import labelMeNav.models as models
from labelMeNav import utils


class Home(View):
    def get(self, request):
        template = loader.get_template('home.html')
        pending_context = models.Image.objects.get_next_pending_images()
        other_constants = {
            "folder_name": constants.LABELME_DIR_NAME,
            # "default_domain": request.build_absolute_uri()
            "default_domain": "http://vm041.bridges.psc.edu/"
        }

        context = {**pending_context, **other_constants}

        return HttpResponse(template.render(context, request))

    post = get


class UpdateDb(View):
    @staticmethod
    def update_image_list_data():
        files = set(utils.get_image_name_list())

        already_exist = set(models.Image.objects.filter(image_name__in=files).values_list('image_name', flat=True))
        to_create = files - already_exist
        for item in to_create:
            models.Image.objects.create(image_name=item)

        return to_create

    def get(self, request):
        created = list(self.update_image_list_data())
        return JsonResponse(created, safe=False)
