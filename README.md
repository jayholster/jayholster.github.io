# Topic Modelling Leonard Bernstein's Young People's Concerts Transcripts

## Data Collection
### Part 1: Mining HTML data

36 out of 53 transcripts for the Young People's Concerts (YPCs) were available on the Leonard Bernstein Office's online archives. I used the beautiful soup and requests python libraries in order to read in all of the html at each respective url. Once each website is read into python, each transcript was appended to the list 'alltranscripts' for futher processing. 

```python
import requests 
from urllib import request, response, error, parse
from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd
import re

urls = ["https://leonardbernstein.com/lectures/television-scripts/young-peoples-concerts/what-does-music-mean",
        "https://leonardbernstein.com/lectures/television-scripts/young-peoples-concerts/what-is-american-music"]
#You can list as many urls here as you like. 

alltranscripts = [] #serves as a placeholder for processed data

# This function calls each url in the list of urls, reads the html data using requests and the parser 'lxml'. 
for url in urls: 
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "lxml")
    transcripts = soup.find_all(class_='col3')
    for item in transcripts:
        alltranscripts.append(item.text)

```

### Part 2: Finding missing data

----
### Purpose Statement

I'm attached to the idea that artificial intelligence will become a core component to social science research in the coming years. In my own work, I use natural language processing tools in order to make meaning out of huge datasets. The tools we use to understand the world are already more powerful and more accessible than ever. Researchers, teachers, and the like should feel empowered to identify and use statistical thinking in the pursuit of some objective truth. 

