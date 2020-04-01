# It's not median household income, unemployment, or population density, coronavirus spread is tied to the size of a county's workforce.

By Jacob Holster, March 29th, 2020

On Friday March 27th the New York Times published coronavirus case data for all U.S. counties. The full article is linked at the bottom of this post. They released two datasets, one for state level, and one for county level. For this research question, we're going to be zooming in on the county level dataset. Variables at the county level, with examples, are:

- date: 2020-01-21
- county: Snohomish
- state: Washington
- fips: 53061
- cases:1
- deaths:0

Federal Information Processing Standard Codes, also known as FIPS codes, are assigned to regions in the United States. Each county has it's own unique FIPS code, and these are across government agencies to keep track of that data points each county creates in a given circumstance.

There were four exceptions in the nytimes dataset. 

### New York City
The New York City boroughs; New York, Kings, Queens, Bronx and Richmond counties were assigned to a single area called New York City. As the individual cases in NYC cannot be tied to specific counties, they were omitted from analysis. 

### Kansas City, Mo.
Kansas City has four overlapping counties (Cass, Clay, Jackson and Platte). A row in the dataset for Kansas City cases and deaths was removed as it contained duplicate data from these four counties. 

### Joplin, Mo.
Joplin is reported separately from Jasper and Newton Counties, and was omitted from analysis due to inconsistencies with MHI figures.

### Chicago, Il.
All Chicago cases have been counted within Cook county, and will be analyzed as such. 

## Employment Data
The United States Department of Agriculture keeps track of data regarding poverty and employment at the county level. They publish their own datasets with FIPS labelled counties, which have been updated as recently as 2018, and 2013 in some cases. The variables of interest in this dataset are below, with example data:

- Rural_urban_continuum_code_2013: 1
- Civilian_labor_force_2018: 430,470	
- Employed_2018: 414,289
- Unemployed_2018: 16,181
- Unemployment_rate_2018: 3.8%
- Median_Household_Income_2018: $87,096	
- Med_HH_Income_Percent_of_State_Total_2018: 117.7%

The rural/urban continuum code represents of the population density on a scale from 1-9. Labor numbers are recorded using data from a 2018 study. The variable 'Med_HH_Income_Percent_of_State_Total_2018' should be interpreted as a percentage distance away from the mean. A datapoint of 117.7 indicates a MHI of 17.7% higher than the state average. 

## Joining Tables
Now that two viable data sources have been identified, they must be joined into one dataset. Here I'm using the pandas library in python to read in and clean employment data from the USDA. The code below reads in the dataset and skips the first four rows, which did not contain any useful data.

```python
import pandas as pd
df = pd.read_csv('countydata.csv', skiprows=4) 
df = df[['FIPS','State','Area_name','Rural_urban_continuum_code_2013','Urban_influence_code_2013','Civilian_labor_force_2018', 'Employed_2018', 'Unemployed_2018', 'Unemployment_rate_2018', 'Median_Household_Income_2018', 'Med_HH_Income_Percent_of_State_Total_2018']]
df = df[df['Rural_urban_continuum_code_2013'].notna()]
df = df[df['Median_Household_Income_2018'].notna()]
df = df.reset_index()
```
I also read in the nytimes dataset 'us-counties.csv'. Here we are renaming the 'fips' column to mach the uppercase titled 'FIPS' column in the USDA dataset. This will allow us to join the tables on this unique column. In the last two lines of code, I drop rows that do not contain a FIPS code, and convert FIPS codes to type integer in preparation for merging. The FIPS codes were initially in float format (ex. 1111.0), and this code gets rid of the decimal at the end (ex. 1111). 

```python
df1 = pd.read_csv('us-counties.csv')
df1 = df1.rename(columns={"fips": "FIPS"})
df1 = df1[df1['FIPS'].notna()]
df1.FIPS = df1.FIPS.astype(int)
```
Now we merge the nytimes data (df1) with the USDA data (df) on the column titled 'FIPS'. The data are grouped by state and county in alphabetical order. The way this dataset works, counties update their most recent numbers. The function '.max()' is applied to columns in the final dataset, as we only want the highest number of reported cases and deaths per county. We then drop columns that are not useful in our final dataframe, and preview the data. 

```python
data = df1.merge(df, on='FIPS', how='inner', suffixes=('_1', '_2'))
data = data.groupby(['state','county']).max()
data = data.drop(['FIPS','index','Area_name','State'], axis=1)
data.to_csv('covidmhi.csv')
```
![Screenshot of Dataset](https://i.imgur.com/rbgJuVm.png)

Upon merging some formatting issues come up. My intention is to run multiple linear regressions, which need float values. Presently, our MHI and other employment data are formatted as such: $40,000, and python needs to read it as 40000.0. Below, I tag the columns that need to be reformatted and apply the necessary transformations. I created a new column titled 'MHIPercDiff' to represent mean differences with state averages by county, and converted the unemployment rate from a percentage (3.6%) to a decimal (.036). 

```python
cols = ['Civilian_labor_force_2018', 'Employed_2018', 'Unemployed_2018', 'Unemployment_rate_2018', 'Median_Household_Income_2018', 'Med_HH_Income_Percent_of_State_Total_2018']
data[cols] = data[cols].replace({'\$': '', ',': ''}, regex=True)
data[cols] = data[cols].apply(pd.to_numeric, axis=1)
data['MHIPercDiff'] = (data['Med_HH_Income_Percent_of_State_Total_2018'] - 100)
data['Unemployment_rate'] = (data['Unemployment_rate_2018']/100)
```
## Constructing Models

For this analysis, we will be using the sklearn and statsmodels python libraries to compute a multiple linear regression, where the number of cases and deaths are predicted by the employment factors. Backwards stepwise eliminations within the model were made for high p values for the individual variable, and changes in the r-squared values. The final model is seen below, where cases are predicted by the variable 'Civilian Labor Force'.    

```python
from statsmodels.formula.api import ols
casemodel = ols('cases ~ Civilian_labor_force_2018', data=data)
casemodel = casemodel.fit()
casemodel.summary()
```
The same technique was applied to the death model, where county unemployment rate remained significant. 

```python
deathmodel = ols('deaths ~  Civilian_labor_force_2018, data=data)
deathmodel = deathmodel.fit()
deathmodel.summary()
```

### Results 
The output for the final case model, followed by the final death model:
![case OLS Output](https://imgur.com/WF2g9Ii.png)
![death OLS Output](https://imgur.com/zKjv7Gb.png)

To get this visualization I used the matplotlib package, and the code below. 

```python
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_regress_exog(casemodel, "Civilian_labor_force_2018", fig=fig)
```
![regressionvisualization](https://imgur.com/pOGjavM.png)

### Interpretation
Based on these results, we can determine that unemployment and median household income are not major players when it comes to the numbers of cases and deaths. Using both models as evidence, the important factor to predict case diagnoses or death is the number of workers in the county. 

To what extent?

R-squared can be interpreteated as the the percent of variation explained by the described model. Simplicity is beneficial in these findings. The R-squared values, always between 0 and 1 decreased by no more than .02 during the course of model pruning. The higher the R-squared value, the stronger the relationship in the data, the more confidence the statistician can have in their hypothesis testing decisions.

Model Equations
- +1 Case = -6.071 + 0.0004(Number of Civilian Workers)
- +1 Death = -0.0383 + .000006(Number of Civilian Workers)

The case model held a strong R-squared value of .49, while the death model's R-squared value was .31. This indicates that the variation in the number of civilian workers can be modeled to describe 49% of the spread in cases, and 31% of the resulting deaths. 0.0004 represents the number of civilian workers increase that fits with the one additional case held constant on the left side of the equation. (Edited, removed incorrect interpretation, see updates below)

Other factors are less important, as far as the model is concerned. However, the lack of significant relationships, for instance within the rural-urban continuum codes, and the urban influence codes, each had no bearing on the change in cases and are interestingly absent in effect. Additionally, median household income had little impact on the strength of any model. The effect of the size of the civilian workforce is isolated within this dataset. 

There is a strong likelihood that the size of a county's civilian workforce has an adverse impact on the spread of cases locally. Conversely, population density, closeness to municipalities, and median household income have little to no effect on the number of diagnosed cases. 

There remains an opportunity to improve the dataset and the subsequent analysis through the joining of datasets with 'FIPS' codes at the county level. I believe it's important to analyze data at the county level because of the unpredicitable nature of the spread of COVID-19. We should be looking for patterns in small places to ascertain our next steps as a society. 

The data are updated daily, and are currently representative of reports as of March 27th. I'll continue updating the model with new data to see if the effect holds. If this trend is established and verified by new data on a continual basis, and compared against other possible confounding variables, this may prove to be a clue towards understanding the patterns of early spread. 

# New Data from March 28th.

After rerunning models using the updated nytimes county dataset, the r-squared values increased; case model moved up to .50, death model up to .34. The beta weights increased, corrected equations are below. I also just caught a mistake in my initial analysis. I claimed that .0004 indicates that 4000 workers equal one case. In fact, every one unit increase in civilian workers, the number of cases will increase by .0004. Therefore we must compute the number of civilian workers needed to increase one case. To do this, we simply divide the increase in case target variable 1/.0004 = 2500. In other wordes, the coefficient increase of .0004 (per worker) tells us that for every 2500 workers, you can expect one case. 

- +1 Case = -5.3972 + 0.0005(Number of Civilian Workers): 1 case per 2000 workers
- +1 Death = -0.0539 + .000007(Number of Civilian Workers): 1 death per 142,857 workers

# New Data from March 30th.
The effects of civilian labor force are still increasing, while the effects of MHI, urban influence, population densitiy, and unemployment are null. 

- +1 Case = -6.6171 + 0.0006(Number of Civilian Workers): 1 case per 1666 workers
- +1 Death = -0.0434 + .000009(Number of Civilian Workers): 1 death per 111,111 workers

Next steps: See if predictions line up with numbers by county, and look for disconfirming cases. 

---
- https://www.nytimes.com/article/coronavirus-county-data-us.html
- https://www.ers.usda.gov/data-products/county-level-data-sets/
- https://github.com/nytimes/covid-19-data

---

# Dynamic Topic Modeling Leonard Bernstein's Young People's Concerts

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


Blei, David M; Lafferty, John D (2006). Dynamic topic models. Proceedings of the ICML. ICML'06. pp. 113–120. doi:10.1145/1143844.1143859. ISBN 978-1-59593-383-6.

---
### Personal Statement

I'm attached to the idea that artificial intelligence will become a core component to social science research in the coming years. In my own work, I use natural language processing tools in order to make meaning out of large datasets. The tools we use to understand the world are already more powerful and more accessible than ever. Researchers, teachers, and the like should feel empowered to identify and use statistical thinking alongside contemporary tools of analysis in the pursuit of some objective truth. 

