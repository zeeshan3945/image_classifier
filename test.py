# This file is used to verify your http server acts as expected
# Run it with `python3 test.py``

import requests
import base64

path = '/Users/zeeshan/Documents/Qdrant-App/coin_images/1695.jpg'
image_file = open(path, 'rb')
image_data = base64.b64encode(image_file.read()).decode('utf-8')

# res = requests.post('http://localhost:8000/', json = {"prompt" : image_data})

# print(res.json())
# docker build -t image_search_feature_extractor .   
# docker run -p 8000:8000 --gpus all image_search_feature_extractor


import banana_dev as banana

api_key = "8d46ac18-5d07-4fa0-a9a8-6f825e1a6a95"
model_key = "0f8c0bdd-142e-4597-9cd0-4ff57c6ce4f9"
model_inputs = {"prompt" : image_data} # anything you want to send to your model

out = banana.run(api_key, model_key, model_inputs)

print(out)