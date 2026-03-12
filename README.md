# Django PDF Chat with Google Gemini (RAG)

A simple and powerful Django-based web application that allows users to upload PDF documents and chat with their content using Retrieval-Augmented Generation (RAG). Powered by **LangChain** and **Google Gemini AI**.

## 🚀 Features

- **PDF Upload**: Easily upload and manage multiple PDF documents.
- **RAG Architecture**: Uses LangChain to split, embed, and store document context for accurate AI responses.
- **Google Gemini Integration**:
  - **Embeddings**: `models/gemini-embedding-001`
  - **Chat LLM**: `models/gemini-flash-latest`
- **Vector Store**: Local **FAISS** index for fast and efficient retrieval.
- **Modern UI**: Sidebar for document selection and a real-time chat interface.

## 🛠️ Prerequisites

- Python 3.10 or higher
- A Google Gemini API Key (get one at [aistudio.google.com](https://aistudio.google.com/))

## 📦 Installation & Setup

1. **Clone the repository** (or navigate to the project folder).

2. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**:
   Create a `.env` file in the root directory (already initialized if you followed the setup):
   ```env
   GOOGLE_API_KEY=your_actual_gemini_api_key_here
   ```

5. **Run Migrations**:
   ```bash
   python manage.py makemigrations
   python manage.py migrate
   ```

6. **Start the server**:
   ```bash
   python manage.py runserver
   ```

7. **Access the app**:
   Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

## 📂 Project Structure

- `chat/`: Main Django app containing logic for PDF processing and chat.
- `chat/utils.py`: Core LangChain RAG pipeline.
- `vector_stores/`: Local storage for FAISS indices (ignored by git).
- `media/pdfs/`: Uploaded PDF files (ignored by git).
- `.env`: Stores sensitive API keys.

## 🔧 Troubleshooting

- **429 RESOURCE_EXHAUSTED**: You've hit the Google Gemini free tier rate limit. Wait 60 seconds and try again.
- **404 NOT_FOUND**: This usually means the model name or API version is incorrect. The project is currently tuned to `models/gemini-flash-latest`.
- **Vector store not found**: Ensure the PDF status shows as "Completed" in the sidebar before chatting. If it says "Failed", check the server console for errors.

## 📝 License
MIT
