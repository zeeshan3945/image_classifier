import torch
from torchvision import transforms, models
from PIL import Image
import base64
from io import BytesIO


device = "cuda:0" if torch.cuda.is_available() else "cpu"

tr = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
    ])


def init():
    global model

    model = models.vgg16(pretrained=True).to(device)
    model.eval()

def inference(model_inputs:dict) -> dict:
    global model

    prompt = model_inputs.get('prompt', None)
    if prompt is None:
        return {'message': "No prompt provided"}

    image_binary = base64.decodebytes(prompt.encode('utf-8'))
    PIL_image = Image.open(BytesIO(image_binary))
    img_tensor = tr(PIL_image.convert("RGB")).unsqueeze(0).to(device)

    # Pass the image through the model
    output = model(img_tensor)

    # Get the predicted class index
    _, predicted = torch.max(output.data, 1)

    return {"Available Device": device,
        "Predicted Class": predicted.item(),
    }
