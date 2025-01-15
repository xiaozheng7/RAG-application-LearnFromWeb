import os
import openai
from langchain_community.chat_models import ChatOpenAI
from typing import Optional
from dotenv import load_dotenv, find_dotenv
from langchain_community.vectorstores import Chroma
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.prompts import PromptTemplate



_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENROUTER_API_KEY']


def get_embedding_and_save_vectorbase(chunks, persist_directory, embeddings):

    vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=persist_directory
    )

    vectordb.persist()


def build_prompt_and_chat_with_llm(persist_directory, embedding, query):

    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

    docs = vectordb.similarity_search(query,k=5)

    context_chunks = [x.page_content for x in docs]

    prompt_start = (
        "Answer the question based on the context below. If you don't know the answer based on the context provided below, just respond with 'I don't know' instead of making up an answer. Return just the answer to the question, don't add anything else. Don't start your response with the word 'Answer:'. Make sure your response is in markdown format\n\n"+
        "Context:\n"
    )
    prompt_end = (
        f"\n\nQuestion: {query}\nAnswer:"
    )

    prompt = ""
    for i in range(1, len(context_chunks)):
        if len("\n\n---\n\n".join(context_chunks[:i])) >= 2000:
            prompt = (
                prompt_start +
                "\n\n---\n\n".join(context_chunks[:i-1]) +
                prompt_end
            )
            break
        elif i == len(context_chunks)-1:
            prompt = (
                prompt_start +
                "\n\n---\n\n".join(context_chunks) +
                prompt_end
            )
            
    import requests
    api_key = os.getenv('OPENROUTER_API_KEY')

    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    messages.append({"role": "user", "content": prompt})
    
    headers = {
    'Authorization': f'Bearer {api_key}',
    'Content-Type': 'application/json'
    }

    data = {
    'model': "meta-llama/llama-3.2-3b-instruct:free", # alternative: "google/gemini-2.0-flash-exp:free",  
    'messages': messages,
    'temperature': 0, 
    'max_tokens': 2000
    }
    
    response = requests.post('https://openrouter.ai/api/v1/chat/completions', headers=headers, json=data)

    response_json = response.json()
    completion = response_json["choices"][0]["message"]["content"]
    return completion
