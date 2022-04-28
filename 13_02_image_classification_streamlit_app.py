from requests.models import MissingSchema
import streamlit as st
import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
import requests
from io import BytesIO

# Create application title and file uploader widget.
st.title("OpenCV Deep Learning based Image Classification")


@st.cache(allow_output_mutation=True)
def load_model():
    """Loads the DNN model."""

    # Read the ImageNet class names.
    with open('classification_classes_ILSVRC2012.txt', 'r') as f:
        image_net_names = f.read().split('\n')

    # Final class names, picking just the first name if multiple in the class.
    class_names = [name.split(',')[0] for name in image_net_names]

    # Load the neural network model.
    model = cv2.dnn.readNet(
        model='DenseNet_121.caffemodel',
        config='DenseNet_121.prototxt',
        framework='Caffe')
    return model, class_names


def classify(model, image, class_names):
    """Performs inference and returns class name with highest confidence."""

    # Remove alpha channel if found.
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    # Create blob from image using values specified by the model:
    # https://github.com/shicai/DenseNet-Caffe
    blob = cv2.dnn.blobFromImage(
        image=image, scalefactor=0.017, size=(224, 224), mean=(104, 117, 123))

    # Set the input blob for the neural network and pass through network.
    model.setInput(blob)
    outputs = model.forward()

    final_outputs = outputs[0]
    # Make all the outputs 1D.
    final_outputs = final_outputs.reshape(1000, 1)
    # get the class label
    label_id = np.argmax(final_outputs)
    # Convert the output scores to softmax probabilities.
    probs = np.exp(final_outputs) / np.sum(np.exp(final_outputs))
    # Get the final highest probability.
    final_prob = np.max(probs) * 100.
    # Map the max confidence to the class label names.
    out_name = class_names[label_id]
    out_text = f"Class: {out_name}, Confidence: {final_prob:.1f}%"
    return out_text


def header(text):
    st.markdown(
        '<p style="background-color:#0066cc;color:#33ff33;font-size:24px;'
        f'border-radius:2%;" align="center">{text}</p>',
        unsafe_allow_html=True)


net, class_names = load_model()

img_file_buffer = st.file_uploader("Choose a file or Camera", type=['jpg', 'jpeg', 'png'])
st.text('OR')
url = st.text_input('Enter URL')

if img_file_buffer is not None:
    # Read the file and convert it to opencv Image.
    image = np.array(Image.open(img_file_buffer))
    st.image(image)

    # Call the classification model to detect faces in the image.
    detections = classify(net, image, class_names)
    header(detections)

elif url != '':
    try:
        response = requests.get(url)
        image = np.array(Image.open(BytesIO(response.content)))
        st.image(image)

        # Call the classification model to detect faces in the image.
        detections = classify(net, image, class_names)
        header(detections)
    except MissingSchema as err:
        st.header('Invalid URL, Try Again!')
        print(err)
    except UnidentifiedImageError as err:
        st.header('URL has no Image, Try Again!')
        print(err)
