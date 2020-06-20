# Clustered Dynamic Topic Modeling Leonard Bernstein's Young People's Concerts
Updated on April 24th, 2020
![What Does Music Mean?](https://bernstein.classical.org/wp-content/uploads/2018/02/GettyImags-53027946_master_metaLR.jpg)

Leonard Bernstein’s Young People’s Concerts reached a wide audience as a television series from the 1950s to the 1970s. There are vast data available regarding the Young People’s Concerts during Bernstein’s tenure as music director, however, these data have yet to be analyzed using exploratory data analysis techniques. Given the volume of the data, rhetorical patterns might be identified to bolster our present perception of Leonard Bernstein as a Music Educator on National television.

What follows is an attempt to engage in historical research with dyanmic topic modeling, which can be used to analyze change in unobserved topics over time within a set of documents (Blei & Lafferty, 2006). An exploratory investigation on the teachings of Leonard Bernstein may reveal connections, or disconnections, between the past and today. Either outcome would be of interest to advocates of music education and musicologists.

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
After concatenating all primary sources, I needed to format the data for time-based inquiry. To make it easier to label the data with the appropriate episode title and airdate, I converted transcript data (bernstein.txt) to a list of sentences, creating a new .csv file. Back to python for now!

```python
# read the txt file, and split every line at the character '.'. Then append sentences to list 'string'. 
string = []    
with open("bernstein.txt", "r") as f: 
    full_text = f.read()
    for l in re.split(r"(\. )", full_text):
        if l != ".":
            string.append(l + ". ")

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

## Part 2: Local Topic Modeling 
In part 1, I converted 50 manuscripts into a single corpus of 4,765 sentences. In part 2, I will do the following: 
- split data into years for local analysis
- run topic models by year 
- rejoin data into single spreadsheet for global analysis. 

I chose to split the data into years for two reasons: 1) Concert cycles, and all programs that the New York Philharmonic conduct are often planned for as seasons which typically align with the juilan calendar. This allows for changes to appear in the final models in conjunction with programmatic changes that may impact production from year to year, and 2). year as a delimiter leads to cleaner, more interpretable results. 

To do this, I split dataset from part 1 on the column 'Airdate', where I kept values that corresponded to each year, and created a new dataframe for those values. I repeated this process until I had a dataframe for each year. 

```python
df1 = df[df['Airdate'].str.contains("58")]
```
I am interested to see the content and number of topics for each year. I used genism and the pyLDAvis packages for this task.

```python
import re
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
%matplotlib inline

# ntlk for stopwords
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
```
The next step is to convert the data for the year 1958 to a list of strings. Then I used re to remove new line characters. 

```python
# Convert to list 
df1 = df1.astype(str)
data = df1.sentence.values.tolist()

#Remove new line characters
data = [re.sub('\s+', ' ', sent) for sent in data]
data = [re.sub('\\n', ' ', sent) for sent in data]
data = [re.sub('\\t', ' ', sent) for sent in data]
```
Then, I assigned functions to: 
- retain sentences as a list of comma separated words, eventually passed to the object 'data_words'.
- remove stopwords
- make bigrams
- make trigrams
- lemmatize text

```python
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out
    
data_words = list(sent_to_words(data))
```
The next step in the creation of a topic model is to build bigram and trigram models. Bigrams are sets of two words often found close to each other in the corpus, tri-grams apply the same concept to sets of three words. To do this, I used genism models as seen below. 

```python
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)
```

The functions come back into play at this point. Here we are removing stopwords, using bigram models, lemmatizing text, and running the output against a spacy parser, which allocates word to word relationships in preparation for a term-document frequency calculation. 

```python
# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)

# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# python3 -m spacy download en
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
#with open('bernstein.txt', 'w') as f:
nlp

# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)

# Create Corpus
texts = data_lemmatized

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]
```
The next step is to build the first LDA model. Fine tuning models prove to be important for attaining readable output. Since I intend on comparing multiple models at the local and global level, topic coherence and independence need to be achieved for each local model. After running several models, I found that topic independence (no overlapping topics) came about when three topics were assumed for 1958. I use a combination of the perplexity and coherence scores alongside my assessment of human readability to determine the sensibility of the model. The final 1958 model held a coherence score of .37, where readings above .3 were satisfactory for my purposes. 

```python
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=3, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=20,
                                           alpha=.05,
                                           per_word_topics=True)
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]

# Compute Perplexity
print('\nPerplexity: ', lda_model.log_perplexity(corpus))

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)
```
Most importantly, these topics were independent of each other. I checked this using pyLDAvis:

```python
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
vis
```
This code produced the following visualization, where the first topic is highlighted. The histogram on the right shows the most relevant terms for the topic. Topical independence is demonstrated on the intertopic distance map, a transformation of principal component analysis for text-modeling. 

![1958LDA](https://imgur.com/RXOfxrc.png)

Next, we need to grab the sentences and keywords assigned to each topic by the model.

```python
def format_topics_sentences(ldamodel=None, corpus=corpus, texts=data):
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row_list in enumerate(ldamodel[corpus]):
        row = row_list[0] if ldamodel.per_word_topics else row_list           
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)


df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=corpus)

# Formatting
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
df_dominant_topic.head()

#sorting
sent_topics_sorteddf_mallet = pd.DataFrame()
sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')

for i, grp in sent_topics_outdf_grpd:
    sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet, 
                                             grp.sort_values(['Perc_Contribution'], ascending=False).head(2000)], 
                                            axis=0)

# Reset Index    
sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)

# Format
sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Representative Text"]

# Show
dfa = sent_topics_sorteddf_mallet
dfa.to_csv('1958analysis.csv')
```
Here is the output of the first LDA. We have a topic number (0.0 or the first topic), an estimation of the percentage likelyhood that the representative text belongs to this topic (.9955), the top ten keywords for the topic, and the attached sentence. 

![1958LDA](https://imgur.com/Bdpa79f.png)

The same process was conducted on data from 1959 - 1972. And all datasets were merged into one in preparation for global analysis. 

```python
df1 = pd.read_csv('1958analysis.csv')
df2 = pd.read_csv('1959analysis.csv')
df3 = pd.read_csv('1960analysis.csv')
df4 = pd.read_csv('1961analysis.csv')
df5 = pd.read_csv('1962analysis.csv')
df6 = pd.read_csv('1963analysis.csv')
df7 = pd.read_csv('1964analysis.csv')
df8 = pd.read_csv('1965analysis.csv')
df9 = pd.read_csv('1966analysis.csv')
df10 = pd.read_csv('1967analysis.csv')
df11 = pd.read_csv('1968analysis.csv')
df12 = pd.read_csv('1969analysis.csv')
df13 = pd.read_csv('1970analysis.csv')
df14 = pd.read_csv('1971analysis.csv')
df15 = pd.read_csv('1972analysis.csv')


frames = [df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14, df15]
bernwithtopics = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14, df15])
bernwithtopics.to_csv('bernwithtopics.csv')
```

## Part 3: Global Analysis
So far, we have been following the dynamic topic modeling steps using clustered LDA, as illustrated in the figure below (Gropp, Herzog, & Apon, 2016). Now that we have merged all split corpi into a categorized corpus, local topics are categorized into subsets of global topics. To do this, we simply run the LDA process on the global list of sentences, retaining local topic labels. My analysis provides evidence of 10 global topics. I compared keywords and representative text across these topics to construct overarching global categories. 

![ClusteredLDA](https://imgur.com/eadIyDh.png)

The global categories and topics were labelled as follows: 

- Sounds: Orchestration, Breaking the Rules, Endings, Heritage
- Stories: Development, Good vs. Evil, Philosophy, Music and Space
- Audience: The Show, Young people

### Final Data Preparations

I am interested in the sentiment of the text, so I used textblob to create a sentiscore. The sentiscore includes polarity, a measure of positive/negative valence (-1.0 to 1.0), and subjectivity, a 0 to 1 scale measuring the extent to which words indicate a fact or an opinion. I ran the following function on the merged dataset to ascertain polarity and subjectivity scores for each text element. This allowed me to compare sentiment measures over time. 

```python
from textblob import TextBlob
def senti(x):
    return TextBlob(x).sentiment  
    
dfa['senti_score'] = dfa['Representative Text'].apply(senti)
```
The last task to complete before visualization is to split the merged dataframe into three dataframes based on emerging categories. This step helps with plotting tasks, which will be completed in R. 

```python
stories = df[(df.gtopics.isin([0,1,8,9]))]
sounds = df[(df.gtopics.isin([2,4,6,7]))]
show = df[(df.gtopics.isin([5,3]))]
stories.to_csv('finalstories.csv')
sounds.to_csv('finalsounds.csv')
show.to_csv('finalshow.csv')
```
### Visualizing Data in R
I analyzed the global topics respective to their category for polarity and subjectivity, and visualized the trends. 

```R
sounds <- read.csv(file = 'finalsounds.csv')
stories <- read.csv(file = 'finalstories.csv')
show <- read.csv(file = 'finalshow.csv')

library('stringr')
library(ggpubr)
library(ggplot2)

show1 <- ggplot(show, aes(year, polarity)) +
  geom_point(aes(color = global_topics)) +
  geom_smooth(se = TRUE) + 
  theme(legend.position = "none") +
  labs(
    title = paste("The Show")
  )

show2 <- ggplot(show, aes(year, subjectivity)) +
  geom_point(aes(color = global_topics)) +
  geom_smooth(se = TRUE) +
  theme(legend.position = "bottom") +
  theme(legend.title = element_blank(), legend.text=element_text(size=5.5))
  
sounds1 <- ggplot(sounds, aes(year, polarity)) +
  geom_point(aes(color = global_topics)) +
  geom_smooth(se = TRUE) + 
  theme(legend.position = "none") +
  labs(
    title = paste("Sounds")
  )

sounds2 <- ggplot(sounds, aes(year, subjectivity)) +
  geom_point(aes(color = global_topics)) +
  geom_smooth(se = TRUE) +
  theme(legend.position = "bottom") +
  theme(legend.title = element_blank(), legend.text=element_text(size=5.5))
  
stories1 <- ggplot(stories, aes(year, polarity)) +
  geom_point(aes(color = global_topics)) +
  geom_smooth(se = TRUE) + 
  theme(legend.position = "none") +
  labs(
    title = paste("Stories")
  )

stories2 <- ggplot(stories, aes(year, subjectivity)) +
  geom_point(aes(color = global_topics)) +
  geom_smooth(se = TRUE) +
  theme(legend.position = "bottom") +
  theme(legend.title = element_blank(), legend.text=element_text(size=5.5)) 
  
ggarrange(sounds1, stories1, show1, sounds2, stories2, show2)
```

This code creates the figure below, where each dot represents a sentence, and color indicates the global topic the sentence belongs to.

![sent](https://imgur.com/RIHuTpc.png)

To visualize global topic mentions by year, I grouped the data by global_topics and year, where counts for each global topic were recorded, code and figure seen below. 

```R
global <- read.csv(file = 'globalperyear.csv')

ggplot(global, aes(x = year, y = global_topics.1)) +
  geom_smooth(aes(color = global_topics), se = FALSE) +
  labs(
    title = paste("Global Topic Mentions By Year")
  )
```
![mentions](https://imgur.com/23O9XtO.png)

... and lastly the most influential keywords using infranodus (Paranyushkin, 2019): 

![mentions](https://imgur.com/gnNDSM7.png)

This culminates the global stage of analysis. The next stage will be locally based (by year, or other trends) which will then be substantiated by secondary sources. 

### Local Abberations and Trends of Note

Polarity and Subjectivity By Year and Category
- Sounds: dip for both models at 1965
- Stories: subjectivity peak at 1967, stable polarity
- The Show: investigate apparent rise and fall for both models
- Investigate outliers for all models

Global Topic Mentions By Year
- Prevalence of 'breaking the rules'
- Stability of orchestration
- Development, young people, and endings path similarity
- Philosophy and space similarity
- Good vs. evil and heritage similarity

Keyword Nodes
- sentences that use these keywords may merit the examination of a set of local topics

## Part 4: Results

The primary purpose for the investigation of local topics was to use the leading keywords to summarize the content of a given year, whereby changes might be perceived from year to year. Conversely, the presence of common important keywords throughout time point to their consistent use over a large period of the Young People’s Concerts. 
Local LDAs

The individual LDAs yielded 76 aggregated local topics, with topic frequencies ranging from three in 1958, to eight in 1963. Figure x below is a histogram, where topic counts are displayed along with three leading topic keyword for each of the 76 local topics. While the scope of this content analysis does not include a detailed investigation of each local topic, in fact most local topics were not explicitly labeled, both the count of unique topics, and the frequency of the most important keywords reveal the words that were most highly connected to the rest of the corpus. 

![histogram](https://i.imgur.com/sjeLPmp.png)

### Local Keywords
Using infranodus I analyzed the full set of local keywords using 4-grams. Bi-grams are better illustrators of the concept. For a list of words, an algorithm searches for word connections. Bi-grams are used to identify the most frequently repeating two-word sets, which are often of some importance to the overall interpretation of the data. 4-grams apply the same concept to words that are within four words of each other, additionally granting a more powerful measure of word connectedness. This connectedness factor is indicated as betweenness in natural language processing research. The table below contains the top 30 most connected, and correrarly the most frequently used keywords in the merged corpus. 

![topkeywords](https://i.imgur.com/6VXqZda.png)

### Global LDA Results
The dataset fit a model of ten topics, which are described with keywords and examples below. 

**Breaking the rules:** The top ten keywords for this topic were: call, know, way, first, new, come, write, mean, and thing. The rules referenced here are actually the rules of music, broadly defined. Composers would often stretch the rules of composition and conductors the rules of performance; Bernstein found this interesting over time. Here, Bernstein describes the way a melody can be manipulated: “But the remarkable thing is not just that a melody is upside-down like a pancake; its the fact that its upside down, and it sounds wonderful upside-down.”

**Endings:** The leading keywords for this topic were: end, ever, find, little, come, bring, piece, final, almost, and perfect. This topic can be interpreted quite literally, as many of the statements made in regard to endings are about musical endings. An example: “And now, as you listen to the fourth movement, which we are about to play with its triumphant ending, you may find different meanings in this ending‚ the rejoicing after a storm, the joy of climbing a mountain and reaching the top, the joy of winning a game, or passing a tough exam, or being well after a sickness; but to the people of Finland this ending will always mean one thing only: freedom.”

**Heritage:** The keywords for ‘Heritage’ include: full, dance, piece sometimes, simple, century, singe, sort, problem, and fun. This is another topic to be interpreted literally, as Bernstein is referring to the connection between many composers’ heritage to the sounds they compose. For instance: “Because in most countries, the people who live there are descended for hundreds of years from their forefathers, and their forefathers forefathers, who all sang the same little tunes and sort of own them; so when the Russians hear a Tchaikovsky symphony, they feel closer to it than say, a Frenchman does, or than we do.”

**Orchestration:** Keywords for ‘Orchestration’ include: instrument, choose, love, sound, use, jazz, rejection, german, and serious, often referring to the choices composers must make as they are writing music: “Sometimes it takes you days or weeks to make up your mind. Well, imagine how hard it is for a composer to make up his mind and choose ‚ not between two things, like a pair of skates or a bicycle ‚ but among all those instruments to say nothing of the hundreds and millions of possible combinations of all those instruments.” 
	
**Development:** Development was described best by the following ten keywords: theme, note, play, also, sing, movement, first, magic, happen, main, and refer to musical development throughout a piece. Berstein describes the manner in which Tchaikovsky develops a theme: “Then he breaks that in half and develops only that half,[ORCH: Tchaikovsky - Fourth Symphony]and now were down to four notes only,[SING: Tchaikovsky - Fourth Symphony]which hes developing in sequences.[SING: Tchaikovsky - Fourth Symphony]But now it divides again, like an amoeba, and the sequence builds on only the last two notes.[SING: Tchaikovsky - Fourth Symphony] Just that.” 

**Good vs. Evil:** A true storyteller, Bernstein sought out the dichotomy of good vs. evil in life, and demonstrated this with the following keywords: faust, bar, orchestra, life, tell, devil, seem, well, and away. Here, Bernstein describes how conflict can be demonstrated through music: “This painful problem is shown in terms of a conflict, the struggle between Man’s tremendous need for immortality, and his equally strong need to accept the fact that he is mortal.”. 

**Music and Space:** The keywords for this topic were: finally, planet, chapter, space, light, understand, religious, suddenly, call, and give. Stories always exist in some space, whether it be outer space, or an impressionist lens toward a sinking cathedral: “You can see the form of a painting, or a church, more or less all at once because their forms exist in space”. Bernstein frequently uses this topic to illustrate the form of a work in relation to its story-like qualities. 

**Philosophy:** The top ten keywords for ‘Philosophy’ were: man, high, story, minor, be, go, beauty, become, beautiful, and even. Throughout the series, Bernstein uses philosophy as a guide for music listening, but also expounds upon the philosophy that is laden within the music he presents. On the connection between Strauss’s ‘Thus Spoke Zarathustra’ and Nietzsche’s Zoroastrian fable: “The connection is a German philosopher named Nietzsche, sorry about all these names, but they’re necessary if were going to make sense out of all this‚ Friedrich Nietzsche, a highly poetic philosopher who was all the rage in Germany when Strauss was a young man.” 

**Young People:** The leading keywords for the topic ‘Young People’ were: young, name, famous, piece, major, great, old, give, key, and applause. Bernstein referred to young people throughout the series, largely from an inspirational perspective: “They said, What? You’re going to play that long, slow, highbrow music for young people? You’re crazy--they’ll get restless and noisy.” This statement was the antithesis to the central motivation behind the Young People’s Concerts, and Bernstein made clear efforts to include the perspective of young people in the presentation of his concerts, where he might add: “I hope you all find it as much fun as I did when I was your age.”

**The Show:** The top ten keywords for ‘The Show’ were: much, program, strange, piano, study, young_people, concert, year, birthday, and chorus. Within this topic, Bernstein made references to the production of the show, to the writing and planning of the show, sometimes including the context of audience feedback: “Ive received so many letters from you in the television audience expressing disappointment because our final program of last year was not televised due to technical difficulties I wont go into, that I have been persuaded to repeat it and so we are going to open this years series by again discussing the subject ‚ what makes music symphonic?”. 

### Defining Global Categories
After examining the content of each of the topic it was clear that there was some overlap. While there were key differences between ‘Breaking the rules of composition’, and ‘Orchestration’, they were both referring to one universal category which I am calling **‘Sounds’**. ‘Endings’, and ‘Heritage’ were also grouped into the ‘Sounds’ category. 

A separate category was created to address commonalities between ‘Good vs. evil’, ‘Philosophy’, ‘Development’, and ‘Music and Space’, and was designated the name **‘Stories’**. The last two global topics are retained in a third category called **‘The Show’** which refers to comments about the Young People’s Concerts production or Bernstein’s efforts to talk about, and talk directly to young people. 

## Next Steps
Next I will be interpreting the sentiment analysis results, lagely on the outliers, as I am interested in unique cases of high or low subjectivity and polarity. 


---
Blei, David M; Lafferty, John D (2006). Dynamic topic models. Proceedings of the ICML. ICML'06. pp. 113–120. doi:10.1145/1143844.1143859. ISBN 978-1-59593-383-6.

Dmitry Paranyushkin. 2019. InfraNodus: Generating Insight Using Text Network Analysis. In The World Wide Web Conference (WWW ’19). Association for Computing Machinery, New York, NY, USA, 3584–3589. DOI:https://doi.org/10.1145/3308558.3314123

Gropp, Chris, Alexander Herzog, Ilya Safro, Paul W. Wilson and Amy W. Apon. “Scalable Dynamic Topic Modeling with Clustered Latent Dirichlet Allocation (CLDA).” ArXiv abs/1610.07703 (2016)

---

### Personal Statement

I'm attached to the idea that artificial intelligence will become a core component to social science research in the coming years. In my own work, I use natural language processing tools in order to make meaning out of large datasets. The tools we use to understand the world are already more powerful and more accessible than ever. Researchers, teachers, and the like should feel empowered to identify and use statistical thinking alongside contemporary tools of analysis in the pursuit of some objective truth. 

