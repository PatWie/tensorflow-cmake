
import requests
import base64
import json
import argparse


def main(fn):
  endpoint = "http://localhost:8501/v1/models/simple_img_model:predict"

  image_buffer = open(fn, "rb").read()
  image_b64 = base64.b64encode(image_buffer).decode("utf-8")

  headers = {"content-type": "application/json"}
  data = json.dumps({
      "inputs": [
          {"b64": image_b64}
      ]
  })

  result = requests.post(endpoint, data=data, headers=headers)
  response = json.loads(result.text)
  response_b64 = response['outputs'][0]
  response_buffer = base64.urlsafe_b64decode(response_b64.encode("utf-8"))
  open('prediction_rest.png', 'wb').write(response_buffer)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('fn', help='path to img', type=str)
  args = parser.parse_args()
  main(args.fn)
