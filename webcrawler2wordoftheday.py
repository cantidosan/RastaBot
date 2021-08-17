from bs4.element import SoupStrainer
import requests
from bs4 import BeautifulSoup, SoupStrainer
import json
import time
import random


url = 'https://jamaicanpatwah.com/'
page = requests.get(url)

soup = BeautifulSoup(page.content, 'html.parser')
doc_elem = soup.find('div', class_='wft-day')


# print(doc_elem.prettify())

wrd_of_day = doc_elem.dt.dfn.h2.contents[0].strip()
# print(wrd_of_day)

wrd_defn = doc_elem.dd.p.contents[0].strip()
# print(wrd_defn)
