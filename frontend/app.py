import streamlit as st
import requests

st.title("Financial Chatbot")

BACKEND_URL = "http://localhost:8000"

if "token" not in st.session_state:
    st.session_state.token = None

username = st.text_input("Username", "demo")
password = st.text_input("Password", "demo123", type="password")

if st.button("Login"):
    try:
        response = requests.post(f"{BACKEND_URL}/token", data={"username": username, "password": password})
        st.write("**Status:**", response.status_code)
        st.write("**Raw Response:**", response.text[:800])  # Show first 800 chars
        
        if response.status_code == 200:
            try:
                st.session_state.token = response.json()["access_token"]
                st.success("Logged in successfully!")
            except:
                st.error("Backend returned 200 but not valid JSON")
        else:
            st.error(f"Login failed with status {response.status_code}")
    except Exception as e:
        st.error(f"Connection error: {str(e)}")

if st.session_state.token:
    question = st.text_area("Ask financial question", height=100)
    if st.button("Submit"):
        try:
            headers = {"Authorization": f"Bearer {st.session_state.token}"}
            response = requests.post(f"{BACKEND_URL}/chat", json={"question": question}, headers=headers)
            st.write("**Status:**", response.status_code)
            st.write("**Raw Response:**", response.text[:800])
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    st.success(f"**Source**: {data.get('source', 'Unknown')}")
                    st.write(data.get("answer", "No answer"))
                except:
                    st.error("Backend returned 200 but not valid JSON")
            else:
                st.error(f"Chat failed with status {response.status_code}")
        except Exception as e:
            st.error(f"Connection error: {str(e)}")
