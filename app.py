import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Funktion zum Laden des Modells
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('./bootleclassification-model_transferlearningFinetuningAugmentation.keras')
    return model

model = load_model()

st.title('Bildklassifikations-App')
st.write("Lade ein Bild hoch, um es vom Modell klassifizieren zu lassen.")

uploaded_file = st.file_uploader("Wähle ein Bild...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Hochgeladenes Bild', use_column_width=True)
    
    # Bild für die Vorhersage vorbereiten
    def prepare_image(img):
        img = img.resize((150, 150))  # Größe anpassen
        img = np.array(img)
        img = img.astype(np.float32)  # Typkonvertierung zu float32
        img = (img / 127.5) - 1  # Skalierung von (0, 255) zu (-1, 1)
        img = np.expand_dims(img, 0)  # Batch-Dimension hinzufügen
        return img

    prepared_img = prepare_image(image)
    predictions = model.predict(prepared_img)
    class_names = ['Beer Bottle', 'Plastic Bottle', 'Soda Bottle', 'Water Bottle', 'Wine Bottle']
    predicted_class_id = np.argmax(predictions)
    predicted_class_name = class_names[predicted_class_id]

    st.write(f'Vorhersage: Klasse {predicted_class_id} - {predicted_class_name}')
    st.write("Wahrscheinlichkeiten:", predictions.flatten())
else:
    st.info("Bitte lade ein Bild hoch, um fortzufahren.")