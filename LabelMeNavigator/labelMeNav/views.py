from django.http import HttpResponse, JsonResponse
from django.template import loader
from django.views import View

import labelMeNav.constants as constants
import labelMeNav.models as models
from labelMeNav import utils


class Home(View):
    def get(self, request):
        template = loader.get_template('home.html')
        context = {
            "total_count": models.Image.objects.all().count(),
            "color_code": constants.color_code,
            "folder_name": "Birds",
            "images": models.Image.objects.all()[:10],
            # "default_domain": request.build_absolute_uri()
            "default_domain": "http://128.237.138.63/"
        }

        return HttpResponse(template.render(context, request))

    post = get


class UpdateDb(View):

    def update_image_list_data(self):
        files = set(utils.get_image_file_list())

        already_exist = set(models.Image.objects.filter(name__in=files).values_list('image_file', flat=True))
        to_create = files - already_exist
        for item in to_create:
            models.Image.objects.create(name=item)

        return to_create

    def get(self, request):
        created = list(self.update_image_list_data())
        return JsonResponse(created, safe=False)
