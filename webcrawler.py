
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
wrd_defn = doc_elem.dd.p.contents[0].strip()

# =================================================================================

# =================================================================================
# =================================================================================
# =================================================================================

url = "https://jamaicanpatwah.com/dictionary/browse/"
init_urls = [url+alpha for alpha in "BCDEFGHIJKLMNOPQRSTUVWXYZ"]

#url2 = 'https://jamaicanpatwah.com/dictionary/browse/A'
#page2 = requests.get(url2)

#soup3 = BeautifulSoup(page2.content, 'html.parser')
wrd_List = []
wrd_def_List = []
patois_dict = {}


def url_to_soup(url):
    page = requests.get(url)

    soup = BeautifulSoup(page.content, 'html.parser')

    return soup


# We've managed to get all the vocab from the Page however the links for the definitions remain untouched
# TODO
# We need to pair  vocabulary definitions
# We need to capture all the pages of their respective letters
# We need to create dictionary of key-value pairs for vocab to definition
# We need to make sure the program has a cooldown for each ping to the server

def get_wrd_dict(soup):
    # extracts all the words from the current souped page ONLY

    doc_elem4 = soup.find_all('div', class_='col')
    for elem in doc_elem4:
        for text_in_space in elem.find_all("a"):
            # reveals direct link to word def
            # print(text_in_space['href'])
            # wrd_List.append((text_in_space.text.strip()))

            new_soup = url_to_soup(text_in_space['href'])

            doc_elem = new_soup.find_all('dl')
            # print(doc_elem[0].find_all('dd')[1].find('p').text.strip())
            patois_def = doc_elem[0].find_all('dd')[1].find('p').text.strip()
            patois_dict[text_in_space.text.strip()] = patois_def
            #time.sleep(int(random.uniform(40, 100)))

            print(patois_dict)
    return patois_dict


def navigate_pages(initial_url, soup):
    #all_urls = [initial_url]
    get_wrd_dict(soup)
    # extracts all the word's definitions from the current souped page ONLY

    doc_elem4 = soup.find_all(class_='global-pagination')

    for elem in doc_elem4:
        # the number of pages available
        print(elem.find('strong').text.strip())
        # the corresponding links
        for urls in elem.find_all('a')[:-1]:

            # print(url['href'])
            # all_urls.append(urls['href'])
            inner_soup = url_to_soup(urls['href'])
            get_wrd_dict(inner_soup)
            print(patois_dict)


# get_wrd_dict(soup3)
def start():
    # print(init_urls)
    for u in init_urls:

        page = requests.get(u)
        soup3 = BeautifulSoup(page.content, 'html.parser')
        navigate_pages(u, soup3)

        with open(u[-1]+".json", "w") as outfile:
            json.dump(patois_dict, outfile)


# start()
