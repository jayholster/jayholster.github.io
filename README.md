## Topic Modelling Leonard Bernstein's Young People's Concerts Transcripts

# Step 1: Mine html data

There are many tutorials for text mining using beautiful soup. 

```python
import requests
from urllib import request, response, error, parse
from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd
import re

urls = ["https://leonardbernstein.com/lectures/television-scripts/young-peoples-concerts/what-does-music-mean",
        "https://leonardbernstein.com/lectures/television-scripts/young-peoples-concerts/what-is-american-music"]
        
alltranscripts = []

for url in urls:
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "lxml")
    transcripts = soup.find_all(class_='col3')
    for item in transcripts:
        alltranscripts.append(item.text)
```
----
### Purpose Statement

I'm attached to the idea that artificial intelligence will become a core component to social science research in the coming years. In my own work, I use natural language processing tools in order to make meaning out of huge datasets. The tools we use to understand the world are already more powerful and more accessible than ever. Researchers, teachers, and the like should feel empowered to identify and use statistical thinking in the pursuit of some objective truth. 

