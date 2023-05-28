import json
from pathlib import Path

import requests
import streamlit as st

# settings
st.set_page_config(
    page_title="Proficiencies review",
    layout="centered",
    initial_sidebar_state="auto",
)

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

# ui
text_placeholder = (
    "Hello, my name is Alice."
    "I'm calling from Chicago and want to ask some questions.\n"
    "I'm pregnant for 6 months now, but I'm not telling anyone about this.\n"
    "I have periodic headaches.\n"
    "When I work, I feel like my ability to concentrate is being hindered by them.\n"
    "It's already hard to work from 9 to 5 every day, God, and now this.\n"
    "My mom told me about this wonder drug called paracetamol."
    " She assured me that it would help me a lot.\n"
    "I'm not sure if that is okay. "
    "It's not like I'm a specialist in this field or anything so"
    " I decided to call here to be sure just in case.\n"
    "Can I use this medicine safely and will it help me?"
)

with st.form("prof_form"):
    text = st.text_area("Input your question:", value=text_placeholder, height=270)

    with st.sidebar:
        top_k = st.slider("How much results to return:", min_value=0, max_value=10)
        threshold = st.slider(
            "Similarity threshold:", min_value=0.0, max_value=1.0, step=0.05
        )
        return_ner = st.checkbox("Return entities")

    submit_button = st.form_submit_button("Ask")

if submit_button:
    url = "http://localhost:8000/query"
    top_k = None if not top_k else top_k
    myobj = {
        "text": text,
        "topk": top_k,
        "threshold": threshold,
        "return_ner": return_ner,
    }

    x = requests.post(url, json=myobj)

    st.write(x.json())
