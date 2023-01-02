import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from io import BytesIO
import base64

# create streamlit page with title "Potato leaf disease classification"

st.title("Potato leaf disease classification")
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]


def load_model():
    model = tf.keras.models.load_model('1.0/1.0')
    st.write(type(model.summary()))



def image_load_process():
    image = st.file_upload("upload an image")









# image_prep()
# image = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'webp'])
# if image is not None:
#     st.image(image, caption='Uploaded Image.', use_column_width=True)
#     st.write("blight init?")

#     def image_process(image):
#         # rescale = tf.keras.Sequential([
#         # tf.keras.layers.Resizing(256, 256),
#         # tf.keras.layers.Rescaling(1.0/255)

#     # ])
#         image = np.array(image)
#         img_batch = np.expand_dims(image, 0)
#         st.write(img_batch)
#         return img_batch

   

#     def predict(model, image):
#         image_process(image)
#         pred = model.predict(image_process(image))
#         pred_class = CLASS_NAMES[np.argmax(pred[0])]
#         return f'Predicted class is {pred_class}'

#     predict(model, image=image_process(image))

# if image is not None:
#     predict(image, model)
# if image is not None:
#     image = Image.open(image)
#     st.image(image, caption='Uploaded Image.', use_column_width=True)
#     st.write("")
#     st.write("Classifying...")
#     label = model.predict(image)
#     if label == 0:
#         st.write("Healthy")
#     elif label == 1:
#         st.write("Early Blight")
#     elif label == 2:
#         st.write("Late Blight")
#     elif label == 3:
#         st.write("Leaf Mold")
#     elif label == 4:
#         st.write("Septoria Leaf Spot")
#     elif label == 5:
#         st.write("Spider Mites")
#     elif label == 6:
#         st.write("Target Spot")
#     elif label == 7:
#         st.write("Yellow Leaf Curl Virus")
#     else:
#         st.write("Mosaic Virus")