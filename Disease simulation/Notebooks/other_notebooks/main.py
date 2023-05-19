import requests
from requests.models import Response
import random
import networkx as nx
from bs4 import BeautifulSoup
import re
import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def download_page (urlString):
    """Takes an url and returns the name of the page and its html data"""
    url = urlString
    r = requests.get(url)
    name = urlString.split("/")[-1]
    return name, BeautifulSoup(r.text, "lxml")

def convert_letters(string):
    """Unicodes æøå ÆØÅ"""
    string = string.replace("%C3%A6", "æ")
    string = string.replace("%C3%B8", "ø")
    string = string.replace("%C3%A5", "å")
    string = string.replace("%C3%86", "Æ")
    string = string.replace("%C3%98", "Ø")
    string = string.replace("%C3%85", "Å")
    return string

def find_neighbors(urlString, breakpoint = 10):
    """
    Takes an html file as a string and returns a set of the names and the links
    """
    urls = re.findall('a href="/wiki/(.*?)"',urlString)
    link_set = set()
    name_set = set()
    counter = 0
    for elm in urls:
        if counter == breakpoint:
            break
        if re.search(":",elm) != None:
            None
        else:
            link_set.add("http://da.wikipedia.org/wiki/" + convert_letters(elm))
            name_set.add(convert_letters(elm))
        counter += 1
    return link_set, name_set

class node:
    """This is each wikipedia page"""
    def __init__(self, url):
        self.url = url
        self.name, self.html_data = download_page(url)
        self.neighbors_link, self.neighbors_name = find_neighbors(str(self.html_data),10)
        self.touched = False
    def info(self):
        print("Url: {}\nName: {}\nAmount of neighbors: {}\nTouched {}".format(self.url, self.name, len(self.neighbors_link), self.touched))

url = 'https://da.wikipedia.org/wiki/Tamkanin'
Tamkanin = node(url)


#Here we initiate the first connection. 
dictionary = dict()
stack = [Tamkanin]
Tamkanin.touched = True
for elm in Tamkanin.neighbors_link:
    nodeTemp = node(elm)
    dictionary[nodeTemp.name] = nodeTemp
    stack.append(nodeTemp)
    

#Breath first search
counter = 0
while len(stack) > counter:
    if stack[counter].touched == False:
        nodeTemp = node(stack[counter].url)
        for links in nodeTemp.neighbors_link:
            temp = node(links)
            if temp.name not in dictionary:
                dictionary[temp.name] = temp
                stack.append(temp)
        stack[counter].touched = True
        nodeTemp.info()
        
    counter += 1
    if counter == 50:
        break

plt.clf()
G = nx.Graph()

#Here we generate the nx graph
for elm in dictionary.items():
    r = lambda: random.randint(0,255)
    c = '#%02X%02X%02X' % (r(),r(),r())
    G.add_node(elm[0])
    nx.set_node_attributes(G, elm[1] ,"obj")
    for n in elm[1].neighbors_name:
        G.add_edge(elm[0],n,color=c)

        
#Plotting
colors = nx.get_edge_attributes(G,'color').values()

plt.rcParams["figure.figsize"] = (400,400)
nx.draw(G,with_labels = True, edge_color = colors)
plt.savefig("test.png")