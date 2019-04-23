from django.shortcuts import render
from django.http import HttpResponse
from django.conf import settings

from .models import Document

import sys
sys.path.append("..") 
from helper import get_results_for_image

def index(request):
    root_dir = "/Users/kohjingyu/Documents/School/Term7/50.035/Project/lasagna/"

    context = {}

    if request.method == 'POST' and request.FILES['img_path']:
        img_path = str(request.FILES['img_path'])

        doc = Document(docfile=request.FILES['img_path'])
        doc.save()

        preds = get_results_for_image(img_path, root_dir=root_dir)
        formatted_preds = []
        # Format tensor to float
        for pred in preds:
            formatted_preds.append((pred[0].capitalize(), round(float(pred[1]), 3)))
        print(formatted_preds)

        context["image_url"] = settings.MEDIA_URL + "images/" + img_path
        context["predicted_ingredients"] = formatted_preds

    return render(request, 'gui/template.html', context)
