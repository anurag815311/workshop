# 📚 My AI Study Assistant

An interactive AI-powered academic assistant built with **Python, Streamlit, Sentence Transformers, and Google Gemini**. The application uses Retrieval-Augmented Generation (RAG) to retrieve relevant information from a predefined knowledge base and generate context-based answers to student questions.

## ✨ Features

- **Interactive Web Interface:** Simple and user-friendly interface built with Streamlit.
- **Semantic Search:** Uses Sentence Transformers to find documents relevant to a user's question.
- **RAG Pipeline:** Combines document retrieval with Gemini-powered answer generation.
- **Context-Based Responses:** Instructs the language model to answer using the retrieved documents.
- **Retrieved Context Viewer:** Allows users to inspect the documents selected for answering their questions.
- **API Key Input:** Accepts a Gemini API key through a password-masked input field.
- **Error Handling:** Displays helpful messages when an API request or other operation fails.

## 🛠️ Technologies Used

- Python
- Streamlit
- Sentence Transformers
- Google Gemini API
- Git and GitHub

## 📖 Knowledge Base

The current application contains predefined information about:

- **Database Management Systems (DBMS):** ACID properties and vector databases.
- **Computer Organization and Architecture (COA):** Cache memory and Amdahl's law.
- **Campus Policies:** Institutional ID cards and general campus information.

The knowledge base can be expanded by adding more documents to the application's document collection.

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/my-ai-study-assistant.git
cd my-ai-study-assistant
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

**Windows PowerShell**

```powershell
.\venv\Scripts\Activate.ps1
```

**Windows Command Prompt**

```cmd
venv\Scripts\activate.bat
```

**macOS/Linux**

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Configure your Gemini API key

Obtain an API key from [Google AI Studio](https://aistudio.google.com/apikey).

Run the application and enter the key in the sidebar. Never commit your API key to GitHub.

### 6. Run the application

```bash
python -m streamlit run day1.py
```

Open the local URL displayed in your terminal, usually:

