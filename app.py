
import json
import pickle
import zipfile
import shutil
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(
    page_title="AI News Text Generator",
    page_icon="📰",
    layout="centered"
)

BASE_DIR = Path(__file__).parent
MODEL_NAME = "news_text_generation_lstm.keras"
ZIP_PATH = BASE_DIR / "model_backup.zip"


@st.cache_resource
def load_files():
    model_path = BASE_DIR / MODEL_NAME

    # If the model file is missing, extract it from the ZIP
    if not model_path.exists():
        if not ZIP_PATH.exists():
            raise FileNotFoundError(
                "Neither the model file nor model_backup.zip was found."
            )

        with zipfile.ZipFile(ZIP_PATH, "r") as archive:
            model_files = [
                name for name in archive.namelist()
                if name.lower().endswith(".keras")
                and not name.startswith("__MACOSX/")
            ]

            if not model_files:
                raise FileNotFoundError(
                    "No .keras model file was found inside model_backup.zip."
                )

            # Copy the model from the ZIP to a temporary location
            temp_dir = Path(tempfile.gettempdir())
            model_path = temp_dir / MODEL_NAME

            with archive.open(model_files[0]) as source:
                with open(model_path, "wb") as destination:
                    shutil.copyfileobj(source, destination)

    model = tf.keras.models.load_model(str(model_path))

    with open(BASE_DIR / "tokenizer.pkl", "rb") as file:
        tokenizer = pickle.load(file)

    with open(BASE_DIR / "config.json", "r", encoding="utf-8") as file:
        config = json.load(file)

    return model, tokenizer, config


st.title("📰 AI News Text Generator")
st.write(
    "Generate news-style text using an LSTM deep learning model."
)

try:
    model, tokenizer, config = load_files()
    st.success("LSTM model loaded successfully!")

    seed_text = st.text_input(
        "Enter starting words",
        value="the government announced"
    )

    next_words = st.slider(
        "Number of words to generate",
        min_value=5,
        max_value=50,
        value=20
    )

    if st.button("Generate News", type="primary"):
        if not seed_text.strip():
            st.warning("Please enter some starting words.")
        else:
            with st.spinner("Generating news text..."):
                result = seed_text.strip()
                max_length = config["max_sequence_length"]

                for _ in range(next_words):
                    token_list = tokenizer.texts_to_sequences(
                        [result]
                    )[0]

                    token_list = pad_sequences(
                        [token_list],
                        maxlen=max_length,
                        padding="pre",
                        truncating="pre"
                    )

                    prediction = model.predict(
                        token_list, verbose=0
                    )[0]

                    prediction[0] = 0
                    if len(prediction) > 1:
                        prediction[1] = 0

                    word_index = int(np.argmax(prediction))
                    word = tokenizer.index_word.get(
                        word_index, ""
                    )

                    if not word or word == "<OOV>":
                        break

                    result += " " + word

            st.subheader("Generated News Text")
            st.write(result)

            st.download_button(
                "Download Generated Text",
                data=result,
                file_name="generated_news.txt",
                mime="text/plain"
            )

except Exception as error:
    st.error(
        "The model files could not be loaded. "
        "Please check the model ZIP, tokenizer and config files."
    )
    st.caption(f"Error details: {error}")
