import os
from pytube import YouTube
from RAG import get_llm_response
import streamlit as st
from transcription import YouTubeVideo
from langchain_cohere import CohereEmbeddings
from langchain_chroma import Chroma

TRANSCRIPT_FOLDER = './transcript'
def open_video_and_ask_questions(database):
    print("done till here")
    question = st.text_area("Ask a question about the video:", key="question")
    if question:
        response = get_llm_response(database, question)
        st.write(response)
        print("done")
        
def start():
    # Title and description
    st.title("YouTube Video Q&A")
    st.write("Enter a YouTube video URL and ask questions about it.")
    # Input box for YouTube URL
    url = st.text_input("Enter YouTube video URL:", key="url")
    if url:
        yt = YouTubeVideo(url)
        if not os.path.exists(f"{YouTubeVideo.TRANSCRIPT_FOLDER}/{yt.filename}.txt"):
            try:
                print("started")
                yt.transcribe()
                db = yt.load_split_embed(2000, 200)
                print("transcription complete")
            except Exception as e:
                st.write(e)
        else:
            db = Chroma(persist_directory="./chroma_db", embedding_function=CohereEmbeddings())
        open_video_and_ask_questions(db)

start()