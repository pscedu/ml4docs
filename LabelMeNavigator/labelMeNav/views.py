import json

from django.http import HttpResponse, JsonResponse
from django.template import loader
from django.utils.decorators import method_decorator
from django.views import View
from django.views.decorators.csrf import csrf_exempt

import labelMeNav.constants as constants
import labelMeNav.models as models
from labelMeNav import utils


class Home(View):
    def get(self, request):
        template = loader.get_template('home.html')
        pending_context = models.Image.objects.get_next_pending_images()
        other_constants = {
            "folder_name": constants.DATASET_NAME,
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


class GetStatus(View):
    def get(self, request, image_name: str = "") -> JsonResponse:
        image = models.Image.objects.filter(image_name=image_name).first()
        if image:
            return JsonResponse(image.status_dict, safe=False)
        else:
            return JsonResponse({}, safe=False)


@method_decorator(csrf_exempt, name='dispatch')
class SetStatus(View):
    def post(self, request) -> JsonResponse:
        request.POST = utils.clean_post_data(json.loads(request.body.decode('utf-8')))

        stamp = request.POST.get("stamp", False)
        page = request.POST.get("page", False)
        image_name = request.POST.get("image_name", False)

        image = models.Image.objects.filter(image_name=image_name).first()
        if image and (stamp or page):
            if stamp:
                models.Image.objects.filter(image_name=image_name).update(human_labeled_stamps=True)
            if page:
                models.Image.objects.filter(image_name=image_name).update(human_labeled_pages=True)

            image.refresh_from_db()
            return JsonResponse(image.status_dict, status=200, safe=False)
        else:
            # HTTP_412_PRECONDITION_FAILED
            return JsonResponse({}, status=412, safe=False)


class LoadNewAnnotations(View):
    def get(self, request):
        result_dict = utils.machine_labeled_updater()
        return JsonResponse(result_dict, safe=False)
