from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from .forms import ImageUploadForm
import cv2
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim
import imutils
import os

def index(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            original_image = form.cleaned_data['original_image']
            tampered_image = form.cleaned_data['tampered_image']

            # Convert uploaded images to OpenCV format
            original_cv = cv2.imdecode(np.frombuffer(original_image.read(), np.uint8), cv2.IMREAD_COLOR)
            tampered_cv = cv2.imdecode(np.frombuffer(tampered_image.read(), np.uint8), cv2.IMREAD_COLOR)

            # Convert the images to grayscale
            original_gray = cv2.cvtColor(original_cv, cv2.COLOR_BGR2GRAY)
            tampered_gray = cv2.cvtColor(tampered_cv, cv2.COLOR_BGR2GRAY)

            # Compute the Structural Similarity Index (SSIM) between the two images
            (score, diff) = ssim(original_gray, tampered_gray, full=True)
            diff = (diff * 255).astype("uint8")

            # Calculate threshold and contours
            thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)

            # Draw the contours on the tampered image
            for c in cnts:
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(tampered_cv, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # Save images to temporary directory
            fs = FileSystemStorage(location='media')
            original_path = fs.save('original.png', original_image)
            tampered_path = fs.save('tampered.png', tampered_image)

            # Save processed images
            original_pil = Image.fromarray(cv2.cvtColor(original_cv, cv2.COLOR_BGR2RGB))
            tampered_pil = Image.fromarray(cv2.cvtColor(tampered_cv, cv2.COLOR_BGR2RGB))
            diff_pil = Image.fromarray(diff)
            thresh_pil = Image.fromarray(thresh)

            original_pil.save(os.path.join('media', 'processed_original.png'))
            tampered_pil.save(os.path.join('media', 'processed_tampered.png'))
            diff_pil.save(os.path.join('media', 'diff.png'))
            thresh_pil.save(os.path.join('media', 'thresh.png'))

            return render(request, 'analysis/result.html', {
                'original_image': fs.url('processed_original.png'),
                'tampered_image': fs.url('processed_tampered.png'),
                'diff_image': fs.url('diff.png'),
                'thresh_image': fs.url('thresh.png'),
                'ssim_score': score,
            })
    else:
        form = ImageUploadForm()
    return render(request, 'analysis/index.html', {'form': form})
