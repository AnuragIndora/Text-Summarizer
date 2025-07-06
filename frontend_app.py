import streamlit as st
import requests

st.set_page_config(page_title="Text Summarizer", layout="centered")

# Define tab layout
tab1, tab2, tab3 = st.tabs(["📝 Summarizer", "🚀 Train Model", "📘 Guide"])

# Helper function to extract text from uploaded file
def extract_text(uploaded_file):
    try:
        if uploaded_file.type == "text/plain":
            return uploaded_file.read().decode("utf-8")

        elif uploaded_file.type == "application/pdf":
            import fitz  # PyMuPDF
            with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
                return "\n".join(page.get_text() for page in doc)

        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            from docx import Document
            doc = Document(uploaded_file)
            return "\n".join([para.text for para in doc.paragraphs])

        else:
            st.warning("Unsupported file type!")
            return ""
    except Exception as e:
        st.error(f"Failed to read document: {e}")
        return ""

# 📝 TAB 1: Summarizer
with tab1:
    st.title("📝 Text Summarizer")
    st.markdown("Upload a document or paste text to get a summary.")

    upload_option = st.radio("Choose input method:", ["📄 Upload Document", "✍️ Paste Text"])

    input_text = ""

    if upload_option == "📄 Upload Document":
        uploaded_file = st.file_uploader("Upload a file (.txt, .pdf, .docx)", type=["txt", "pdf", "docx"])
        if uploaded_file:
            input_text = extract_text(uploaded_file)
            if input_text:
                st.success("File uploaded and text extracted successfully!")
                st.text_area("Extracted Text", value=input_text, height=250)
    else:
        input_text = st.text_area("Enter your text to summarize", height=300)

    if st.button("Summarize"):
        if not input_text.strip():
            st.warning("Please enter or upload some text.")
        elif len(input_text) > 5000:
            st.warning("Input text is too long (limit 5000 characters). Please shorten it.")
        else:
            with st.spinner("Summarizing..."):
                try:
                    response = requests.post(
                        "http://127.0.0.1:8000/summarize",  # Updated endpoint
                        json={"input_text": input_text}  # Correct key for the request body
                    )
                    result = response.json()

                    if "summary" in result:
                        st.subheader("📄 Summary:")
                        st.success(result["summary"])
                    else:
                        st.error(result.get("error", "Unknown error occurred."))
                except requests.exceptions.RequestException as e:
                    st.error(f"Backend not reachable: {e}")

# 🚀 TAB 2: Train Model
with tab2:
    st.title("🚀 Train Model")
    st.markdown("Click the button below to start training the model.")

    if st.button("Train Model"):
        with st.spinner("Training the model..."):
            try:
                response = requests.get("http://127.0.0.1:8000/train")  # Endpoint to trigger training
                if response.status_code == 200:
                    st.success("Model training completed successfully!")
                else:
                    st.error("Error during training: " + response.json().get("detail", "Unknown error."))
            except requests.exceptions.RequestException as e:
                st.error(f"Backend not reachable: {e}")

# 📘 TAB 3: Guide
with tab3:
    st.title("📘 How It Works & Features")
    st.markdown("""
    ### 🧠 How It Works
    - This app uses a **transformer-based NLP model** to generate summaries.
    - The model is trained or fine-tuned using your dataset.
    - The **FastAPI** backend serves as the inference and training API.

    ### 💡 Features
    - Text summarization using a pre-trained transformer
    - Model training on demand with customizable dataset sizes
    - User-friendly UI built with **Streamlit**
    - Clear API design powered by **FastAPI**

    ### 🔌 API Endpoints (via FastAPI)
    - `POST /summarize` – Generate summary from text
    - `GET /train` – Retrain or fine-tune the summarizer with dataset sizes

    ### 🛠️ Notes
    - Make sure the backend is running at `http://127.0.0.1:8000`
    - Summary length and accuracy depend on the model and training data

    ---
    Made with ❤️ using Streamlit + FastAPI + Transformers
    """)

