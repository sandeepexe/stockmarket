import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from streamlit_lottie import st_lottie
import requests

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

st.set_page_config(
    page_title = "Stock Prediction App",
    page_icon = "📈",
)

st.subheader("Support your intuition🧠 by visualizing Historical Data!")

st.title("Stock Prediction App")

lottie_app1 = load_lottieurl("https://lottie.host/1c171639-c7eb-41f8-8c23-223d8a4265b3/36Fo1aSOI8.json")

st_lottie(lottie_app1, speed = 1, key = "app1", height = 400, width = 400)

if st.button("Predict Stock Trend"):
    switch_page("📈 prediction")