# Dynamic Topic Modeling: Leonard Bernstein's Young People's Concerts

What follows is an attempt to engage in historical research with dyanmic topic modeling, which can be used to analyze change in unobserved topics over time within a set of documents (Blei & Lafferty, 2006).

## Part 1: Building the Primary Source Dataset
### Mining HTML Data

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
    #class = 'col3' refers to the common HTML containers which held all transcript data.  
    for item in transcripts:
        alltranscripts.append(item.text)

```

### Missing Data

I contacted the archivist at the Leonard Bernstein Office to inquire about the missing transcripts. They responded quickly, attaching pdfs of typewritten transcripts. I was able to locate all but three transcripts, which the archivist indicated were not going to be developed into transcripts. 

I converted pdfs to txt files using the script below. While there are pdf parsers available for python, I found the most success using the pdftools library in R.

```R
library(pdftools)
files <- list.files(pattern = 'pdf$')
txttranscripts <- lapply(files, pdf_text)
```

Additionally, I discovered that Jamie Bernstein, daughter of Leonard Bernstein, hosted a Young People's Concert as part of the Bernstein at 100 festival which took place at the University of Colorado Boulder in 2018. After contacting represetentatives of the College of Music, I was able to procure a transcript of this event and add it to my list, resulting in 132 pages of transcripts which included 51,956 words, or 500,827 characters. 

### Labeling the Dataset
After concatenating all primary sources, I needed to format the data for time-based inquiry. To make it easier to label the data with the appropriate episode title and airdate, I converted transcript data (bernstein.txt) to a list of sentences, creating a new .csv file. 

```python
# read the txt file, and split every line at the character '.'. Then append sentences to list 'string'. 
string = []    
with open("bernstein.txt", "r") as f: 
    #full_text = f.read()
    #for l in re.split(r"(\. )", full_text):
      #  if l != ".":
        #    string.append(l + ". ")

#convert list to dataframe 
df = pd.DataFrame(string)
#drop empty rows
df.dropna()
#print to a new csv file
df.to_csv('bernsteinsentences.csv')
```

My data were converted to 4,765 sentences. I wanted to label each sentence by episode title and original airdate, so I cross checked the leading and final sentences of each episode on the original transcripts with the new data frame, and manually labelled the rows. 

![Screenshot of Dataset](https://i.imgur.com/tHbJD9w.png)

More preprocessing and data cleaning to come, but this initial dataset will set me up for topic modeling using Latent Dirichlet Allocation (LDA) and sentiment analysis, the two primary components of my dynamic model analysis. 


Blei, David M; Lafferty, John D (2006). Dynamic topic models. Proceedings of the ICML. ICML'06. pp. 113–120. doi:10.1145/1143844.1143859. ISBN 978-1-59593-383-6.

---
### Personal Statement

I'm attached to the idea that artificial intelligence will become a core component to social science research in the coming years. In my own work, I use natural language processing tools in order to make meaning out of huge datasets. The tools we use to understand the world are already more powerful and more accessible than ever. Researchers, teachers, and the like should feel empowered to identify and use statistical thinking alongside contemporary tools of analysis in the pursuit of some objective truth. 

