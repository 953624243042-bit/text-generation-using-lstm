
import json
import pickle
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(
    page_title="AI News Text Generator",
    page_icon="📰",
    layout="centered"
)

@st.cache_resource
def load_files():
    model = tf.keras.models.load_model(
        "news_text_generation_lstm.keras"
    )

    with open("tokenizer.pkl", "rb") as file:
        tokenizer = pickle.load(file)

    with open("config.json", "r") as file:
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

                    # Ignore padding and OOV tokens
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
        "Check that the model, tokenizer and config files "
        "are uploaded correctly."
    )
    st.caption(f"Error details: {error}")
