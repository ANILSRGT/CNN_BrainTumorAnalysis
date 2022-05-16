import numpy as np
import tensorflow as tf
from PIL import Image
from keras.preprocessing import image


def TestRun(tfliteModelPath: str, imagePath: str, mean: float, variance: float, entropy: float, img_size=(64, 64)):
    img = image.load_img(imagePath)
    tflite_interpreter = tf.lite.Interpreter(model_path=tfliteModelPath)
    tflite_interpreter.allocate_tensors()
    input_details = tflite_interpreter.get_input_details()
    output_details = tflite_interpreter.get_output_details()

    input_img = Image.open(imagePath)
    input_img = input_img.resize(img_size)
    red, green, blue = input_img.split()
    x_matrix = np.array([np.array(blue)], 'f')
    x_matrix = x_matrix.reshape(input_details[0]['shape'])
    print(input_details[0]['shape'])

    tflite_interpreter.set_tensor(input_details[0]['index'], x_matrix)
    tflite_interpreter.invoke()
    y_matrix = tflite_interpreter.get_tensor(output_details[0]['index'])
    print(y_matrix)
