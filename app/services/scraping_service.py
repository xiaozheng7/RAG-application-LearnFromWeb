import requests
from bs4 import BeautifulSoup

def scrape_website(url):
    '''langchain also provides several types of loaders'''
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    text = soup.get_text(separator='\n')
    return text