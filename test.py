import base64
import banana_dev as banana
import time


path = '../Image_Classification/imagenet-sample-images/n01440764_tench.JPEG'

with open(path, "rb") as f:
  image_bytes = f.read()

image_base64 = base64.b64encode(image_bytes).decode("utf-8")

api_key = "8d46ac18-5d07-4fa0-a9a8-6f825e1a6a95"
model_key = "0f8c0bdd-142e-4597-9cd0-4ff57c6ce4f9"
model_inputs = {"prompt" : image_base64} 

start = time.time()
out = banana.run(api_key, model_key, model_inputs)
end = time.time()

print(f"Time taken by Banana.dev: {end - start}")
print(out['modelOutputs'])

