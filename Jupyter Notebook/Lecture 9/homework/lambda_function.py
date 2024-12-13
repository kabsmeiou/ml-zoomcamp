# tflite_runtime and keras-image-helper
import tflite_runtime.interpreter as tflite
from keras_image_helper import create_preprocessor
import numpy as np

interpreter = tflite.Interpreter(model_path='model_2024_hairstyle_v2.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

from io import BytesIO
from urllib import request

from PIL import Image

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

def prepare_input(x):
    return x / 255.0


def predict(url):
    img = download_image(url)
    prepared_img = prepare_image(img, (200, 200))
    x = np.array(prepared_img, dtype='float32')
    X = np.array([x])
    X = prepare_input(X)
    # initialize input
    interpreter.set_tensor(input_index, X)
    # invoke the interpreter
    interpreter.invoke()
    # fetch output
    preds = interpreter.get_tensor(output_index)
    return preds

def lambda_handler(event, context):
    url = event['url']
    result = predict(url)
    return result

