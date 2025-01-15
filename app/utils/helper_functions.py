from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


def chunk_text(text, chunk_size=200):

    documents = [Document(page_content=text)]
    r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size, chunk_overlap=0, separators=["\n\n", "\n", "(?<=\. )", " ", ""] 
)

    chunks = r_splitter.split_documents(documents) 

    return chunks

