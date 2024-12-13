import tflite_runtime.interpreter as tflite
from keras_image_helper import create_preprocessor

classes = [
    'dress',
    'hat',
    'longsleeve',
    'outwear',
    'pants',
    'shirt',
    'shoes',
    'shorts',
    'skirt',
    't-shirt'
]

interpreter = tflite.Interpreter(model_path='clothing-model.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

# check details for input
interpreter.get_input_details()

# preprocessor
preprocessor = create_preprocessor('xception', target_size=(299, 299))

# url = 'http://bit.ly/mlbookcamp-pants'

def predict(url):
    X = preprocessor.from_url(url)

    # initialize input
    interpreter.set_tensor(input_index, X)

    # invoke the interpreter
    interpreter.invoke()

    # fetch output
    prediction = interpreter.get_tensor(output_index)

    # convert to float32
    float_prediction = prediction[0].tolist()

    return dict(zip(classes, float_prediction))

def lambda_handler(event, context):
    url = event['url']
    result = predict(url)
    return result



