import streamlit as st
import requests

st.title("Financial Chatbot")

if st.button("Login (demo / demo123)"):
    st.session_state.token = "demo-token"

if "token" in st.session_state:
    q = st.text_input("Ask financial question:")
    if st.button("Send"):
        r = requests.post("http://localhost:8000/chat", json={"question": q}, headers={"Authorization": f"Bearer {st.session_state.token}"})
        st.write(r.json())
