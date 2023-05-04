import requests
import base64
import time

# before running thsi script, make sure that the application is runnig on local server (run python3 server.py)
path = '../Image_Classification/imagenet-sample-images/n01440764_tench.JPEG'

with open(path, "rb") as f:
  image_bytes = f.read()

image_base64 = base64.b64encode(image_bytes).decode("utf-8")
model_inputs = {"prompt" : image_base64} 

start = time.time()
response = requests.post('http://localhost:8000/', json = model_inputs)
end = time.time()

print(f"Time take by request {end-start} seconds")
print(response.text)