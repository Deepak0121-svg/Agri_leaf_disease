# ====== IMPORT LIBRARIES ======
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

# ====== LOAD THE MODEL ======
model = tf.keras.models.load_model("model/mobilenetv2_LeafDisease_model_trained.h5")

# ====== CLASS LABELS WITH DISEASE MANAGEMENT SUGGESTIONS ======
class_labels = [
    ("Apple___Apple_scab", "Use fungicides, remove infected leaves, and practice crop rotation."),
    ("Apple___Black_rot", "Prune infected areas, use resistant varieties, and apply copper-based fungicides."),
    ("Apple___Cedar_apple_rust", "Remove infected leaves and apply fungicides at early stages."),
    ("Apple___healthy", "Keep your apple trees well-maintained and avoid over-watering."),
    ("Blueberry___healthy", "Ensure proper drainage and apply organic mulch to maintain soil moisture."),
    ("Cherry_(including_sour)___Powdery_mildew", "Use fungicides and prune the tree to improve air circulation."),
    ("Cherry_(including_sour)___healthy", "Ensure proper watering, pruning, and check for any pests."),
    ("Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot", "Use resistant varieties and fungicides."),
    ("Corn_(maize)___Common_rust_", "Apply fungicides during early infection and rotate crops."),
    ("Corn_(maize)___Northern_Leaf_Blight", "Apply fungicides and rotate crops to reduce the spread."),
    ("Corn_(maize)___healthy", "Ensure proper irrigation and soil conditions to keep the plants healthy."),
    ("Grape___Black_rot", "Prune infected areas, apply fungicides, and remove fallen leaves."),
    ("Grape___Esca_(Black_Measles)", "Remove infected vines, apply copper-based fungicides."),
    ("Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Apply fungicides and avoid overhead irrigation."),
    ("Grape___healthy", "Maintain proper soil conditions and remove any dead leaves."),
    ("Orange___Haunglongbing_(Citrus_greening)", "Use disease-free seedlings and apply insecticides for psyllid control."),
    ("Peach___Bacterial_spot", "Prune infected areas and apply copper-based bactericides."),
    ("Peach___healthy", "Water regularly and avoid overcrowding to ensure good airflow."),
    ("Pepper,_bell___Bacterial_spot", "Use resistant varieties and apply copper-based bactericides."),
    ("Pepper,_bell___healthy", "Fertilize and water regularly to ensure healthy growth."),
    ("Potato___Early_blight", "Use fungicides, remove infected leaves, and practice crop rotation."),
    ("Potato___Late_blight", "Use resistant varieties, and apply fungicides and proper irrigation."),
    ("Potato___healthy", "Ensure the plants receive enough sunlight and avoid waterlogging."),
    ("Raspberry___healthy", "Water regularly and mulch to keep the soil moist."),
    ("Rice_BrownSpot", "Apply fungicides and use resistant varieties."),
    ("Rice_Healthy", "Water regularly and avoid overcrowding the plants."),
    ("Rice_Hispa", "Use insecticides or manually remove the pests."),
    ("Rice_LeafBlast", "Apply fungicides and use resistant rice varieties."),
    ("Soybean___healthy", "Fertilize regularly and ensure proper irrigation."),
    ("Squash___Powdery_mildew", "Use fungicides and remove infected leaves to prevent the spread."),
    ("Strawberry___Leaf_scorch", "Reduce watering and provide proper ventilation."),
    ("Strawberry___healthy", "Ensure proper irrigation and protect from pests."),
    ("Tomato___Bacterial_spot", "Use copper-based bactericides and prune affected areas."),
    ("Tomato___Early_blight", "Use fungicides and remove infected leaves."),
    ("Tomato___Late_blight", "Use resistant varieties and apply fungicides."),
    ("Tomato___Leaf_Mold", "Prune for airflow, remove infected leaves, and apply fungicides."),
    ("Tomato___Septoria_leaf_spot", "Remove infected leaves and apply fungicides."),
    ("Tomato___Spider_mites_Two-spotted_spider_mite", "Use miticides and remove heavily affected leaves."),
    ("Tomato___Target_Spot", "Apply fungicides and remove affected leaves."),
    ("Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Use resistant varieties and remove infected plants."),
    ("Tomato___Tomato_mosaic_virus", "Remove infected plants and disinfect tools."),
    ("Tomato___healthy", "Ensure proper watering, soil drainage, and pest management.")
]

class_labels_dict = dict(class_labels)

# ====== STREAMLIT UI ======
st.title("🌿 Leaf Disease Detector")
st.write("Upload a leaf image to detect the disease and get management tips.")

uploaded_image = st.file_uploader("Upload a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_image:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="📷 Uploaded Image", use_column_width=True)

    # ====== PREPROCESS IMAGE ======
    img_array = np.array(image.resize((224, 224))) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # ====== PREDICT ======
    predictions = model.predict(img_array)[0]
    top_3_indices = predictions.argsort()[-3:][::-1]
    top_3_labels = [class_labels[i][0] for i in top_3_indices]
    top_3_probs = [predictions[i] for i in top_3_indices]

    # ====== DISPLAY TOP PREDICTION ======
    top_label = top_3_labels[0]
    top_prob = top_3_probs[0]
    confidence_score = top_prob * 100

    if confidence_score > 75:
        severity = "Mild"
    elif confidence_score > 50:
        severity = "Moderate"
    else:
        severity = "Severe"

    suggestion = class_labels_dict.get(top_label, "No suggestion available.")

    st.subheader("🩺 Prediction Result")
    st.markdown(f"🔍 **Detected Disease:** `{top_label}`")
    st.markdown(f"📊 **Confidence:** `{confidence_score:.2f}%`")
    st.markdown(f"⚠️ **Severity:** `{severity}`")
    st.markdown("🌱 **Disease Management Tip:**")
    st.info(suggestion)

    # ====== COLOR SET FOR CHART ======
    color_list = ["#00a65a", "#f39c12", "#dd4b39"]  # green, yellow-orange, red

    # ====== BAR CHART WITH CONFIDENCE PERCENTAGES ======
    df = pd.DataFrame({
        "Disease": top_3_labels,
        "Confidence": [round(p * 100, 2) for p in top_3_probs]
    })

    fig, ax = plt.subplots()
    bars = ax.barh(df["Disease"], df["Confidence"], color=color_list)
    ax.set_xlabel("Confidence (%)")
    ax.set_title("Top 3 Predictions")
    ax.invert_yaxis()

    # ====== ADD PERCENTAGE ON BARS ======
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height() / 2,
                f"{df['Confidence'][i]}%", va='center')

    st.pyplot(fig)
