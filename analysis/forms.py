from django import forms

class ImageUploadForm(forms.Form):
    original_image = forms.ImageField()
    tampered_image = forms.ImageField()
