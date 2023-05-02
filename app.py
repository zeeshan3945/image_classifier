import torch
from torchvision import transforms, models
from PIL import Image
from io import BytesIO
import base64

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
def init():
    global model
    # Load the pretrained VGG16 model
    print("loading to CPU...")
    model = models.vgg16(pretrained=True)

    # conditionally load to GPU
    if device == "cuda:0":
        print("loading to GPU...")
        model.cuda()
        print("done")

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model

    # Parse out your arguments
    prompt = model_inputs.get('prompt', None)
    if prompt == None:
        return {'message': "No prompt provided"}
    
    tr = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
    ])
    
    image_binary = base64.b64decode(prompt)
    PIL_image=Image.open(BytesIO(image_binary))

    img_tensor = tr(PIL_image.convert("RGB"))

    img_tensor = img_tensor.unsqueeze(0)
    model.eval()
    # Pass the image through the model
    output = model(img_tensor)

    # Get the predicted class index
    _, predicted = torch.max(output.data, 1)

    # Return the predicted array as the response
    return {
                "Predicted Class": predicted.item(),
            }
            