## RAG Chatbot for YouTube videos
### 📍Overview
That chatbot helps you ask questions about the contents of a youtube video just by inputting a link to the video
It uses LangChain to create RAG pipeline, youtube-transcript-api for transcript and openai-whisper for speech-to-text capabilites

## 🚀 Getting Started

**System Requirements:**

  - Python 3.9+
  - Package manager: `pip`
  - LLM service: `Google Gemini`
  
## 🤖 Running Locally

    - ### Clone this repository
      - git clone https://github.com/cs7-shrey/youtube-rag-chatbot.git
    - ### Install Dependencies
      - pip install -r requirements.txt
    - ### Setup GOOGLE_API_KEY and COHERE_API_KEY as environement variables
      - get api key from respective providers
    - ### Run the app
      - python -m streamlit run app.py