import os, shutil
from pytube import YouTube
import whisper
from youtube_transcript_api import TranscriptsDisabled, YouTubeTranscriptApi

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere.embeddings import CohereEmbeddings   
from langchain_chroma import Chroma


def get_video_id(url: str) -> str:
    if "watch?v=" in url:
        lst = url.split("watch?v=")
        new_list = lst[-1].split("&list")
        return new_list[0]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[1]
    else: 
        return ValueError("Invalid YouTube URL")

def clear_transcript():
    folder = YouTubeVideo.TRANSCRIPT_FOLDER
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


class YouTubeVideo:
    TRANSCRIPT_FOLDER = './transcript'
    def __init__(self, URL):
        self.URL = URL
        yt = YouTube(URL)
        audio = yt.streams.filter(only_audio=True).first()
        filename = audio.default_filename
        self.filename = filename
    def transcribe(self):
        yt = YouTube(self.URL)
        audio = yt.streams.filter(only_audio=True).first()
        filename = audio.default_filename
        if not os.path.exists(f"{YouTubeVideo.TRANSCRIPT_FOLDER}/{filename}.txt"):
            clear_transcript()
            try:
                video_id = get_video_id(self.URL)
                transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'en-IN', 'en-US', 'en-UK'])
                transcript = ""
                for i in transcript_list:
                    transcript += i['text']
                    transcript += " "
                with open(f"{YouTubeVideo.TRANSCRIPT_FOLDER}/{filename}.txt", "w") as file1:
                    file1.write(transcript)

            except TranscriptsDisabled:
                whisper_model = whisper.load_model("base")
                audio.download()
                transcript = whisper_model.transcribe(filename, fp16=False)["text"].strip()
                with open(f"{TRANSCRIPT_FOLDER}/{filename}.txt", "w") as file1:
                    file1.write(transcript)
        else:
            with open(f"{YouTubeVideo.TRANSCRIPT_FOLDER}/{filename}.txt", "r") as file:
                transcript = file.read()
        return transcript
    def load_split_embed(self, chunk_size: int, chunk_overlap: int):
        embeddings = CohereEmbeddings()
        loader = TextLoader(f"{YouTubeVideo.TRANSCRIPT_FOLDER}/{self.filename}.txt")
        text_documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        split_docs = text_splitter.split_documents(text_documents)
        db = Chroma.from_documents(split_docs, embeddings, persist_directory="./chroma_db")
        return db


