import streamlit as st
import PyPDF2
import json
import requests

st.title("Extract Information Curriculum")

def extract_text_from_pdf(uploaded_file):
    text = ""
    
    reader = PyPDF2.PdfReader(uploaded_file)
    for page in reader.pages:
        text += page.extract_text()
    return text

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    st.write("Processing file...")
    text = extract_text_from_pdf(uploaded_file)
    

    try:
        

        url = "http://127.0.0.1:8000/prompt"

        payload = json.dumps({
                "promptText": text
                
                })
        headers = {
        'Content-Type': 'application/json'
        }
        
        response = requests.request("POST", url, headers=headers, data=payload, verify=False)
        response_json=json.loads(response.text)
        
        formatted_json = json.dumps(response_json, indent=4, ensure_ascii=False)

        
        st.text_area("Text Output", formatted_json, height=500)
        st.download_button(
                                label="Download JSON",
                                data=formatted_json,
                                file_name="response.json",
                                mime="application/json"
                            )                
    except Exception as e:
        print(e)

