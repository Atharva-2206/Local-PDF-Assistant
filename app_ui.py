import streamlit as st
import requests
import uuid

# Define API URLs
PDF_PROCESS_API = "http://localhost:8000/process-pdf/"
CHAT_API = "http://localhost:8001/chat/"

# Streamlit App
st.title("Local PDF Assistant")

# Section for PDF upload and processing
st.header("Upload a PDF")
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file is not None:
    # Save the uploaded file locally
    transaction_id = str(uuid.uuid4())
    file_path = f"./backend/data/uploads/{transaction_id}_{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Show file upload status
    st.success(f"File uploaded: {uploaded_file.name}")

    # Call the PDF processing API to process the file
    try:
        response = requests.post(PDF_PROCESS_API, files={"file": uploaded_file})
        if response.status_code == 200:
            st.success("PDF processed successfully!")
            st.text(f"Transaction ID: {response.json()['transaction_id']}")
        else:
            st.error(f"Failed to process PDF: {response.text}")
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to the PDF processing API: {e}")

# Section for Chat with PDF
st.header("Chat with the Processed PDF")

# Input for the question
transaction_id_input = st.text_input("Enter the Transaction ID", "")
question_input = st.text_area("Ask a question based on the PDF content:")

if st.button("Ask Question"):
    if transaction_id_input and question_input:
        try:
            # Make the request to the chat API
            response = requests.post(
                CHAT_API,
                json={
                    "transaction_id": transaction_id_input,
                    "question": question_input
                }
            )

            if response.status_code == 200:
                st.success("Answer: ")
                st.write(response.json()["answer"])
            else:
                st.error(f"Error: {response.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to the chat API: {e}")
    else:
        st.error("Please provide both Transaction ID and Question.")