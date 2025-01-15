from . import api_blueprint
from flask import request, jsonify, abort, current_app
import logging
from app.services import scraping_service, llm_and_embedding_service
from app.utils.helper_functions import chunk_text

index_name = 'index119'
persist_directory = f'docs/chroma/{index_name}/'

@api_blueprint.route('/')
def hello_world():
    return 'Hello, World!'


@api_blueprint.route('/embed-and-store', methods=['POST'])
def embed_and_store():
    url = request.json['url']
    url_text = scraping_service.scrape_website(url)
    chunks = chunk_text(url_text)
    embeddings = current_app.embeddings
    llm_and_embedding_service.get_embedding_and_save_vectorbase(chunks, persist_directory, embeddings)
    response_json = {
        "message": "Chunks embedded and stored successfully"
    }
    return jsonify(response_json)

@api_blueprint.route('/handle-query', methods=['POST'])
def handle_query():
  embeddings = current_app.embeddings
  question = request.json['question']
  answer = llm_and_embedding_service.build_prompt_and_chat_with_llm(persist_directory, embeddings, question)
  return jsonify({ "question": question, "answer": answer })    