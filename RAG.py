import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere.embeddings import CohereEmbeddings   
from langchain_chroma import Chroma

GOOGLE_API_KEY = os.environ['GOOGLE_API_KEY']

def get_llm_response(db, query: str) -> str:
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the giver question based on the context below. If you don't know the answer, just say "I don't know"
        <context>
        {context} 
        </context>
        Question: {question}
        """
    )
    output_parser = StrOutputParser()
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", api_key = GOOGLE_API_KEY, max_output_tokens=2048)
    docs = db.similarity_search(query)
    # print(docs[0])
    retriever = db.as_retriever()
    print(retriever.invoke(query))
    retrieval_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | output_parser
    )
    response = retrieval_chain.invoke(query)
    return response


if __name__ == "__main__":
    main()

