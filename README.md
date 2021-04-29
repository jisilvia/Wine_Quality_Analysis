# Wine Quality Analysis - Wine Features and their Influence on Ratings
<p align="center">
  <img width="auto" height="auto" src="https://image.freepik.com/free-vector/illustration-people-drinking-wine_53876-37281.jpg">
</p>
Whether it is for resale or consumption, wines should be chosen carefully as many factors can affect its taste, regardless if they are biased or not. To provide guidance to both retailers and consumers, this paper will determine which factors have the highest influence on a wine’s overall rating. 


# Project Description

The main objective of this analysis is to determine the most important factors that affect a wine’s overall rating. The factors and ratings were obtained from a collection of wine reviews by sommeliers at Wine Enthusiast, which was then analyzed to identify which factors had the highest importance and which factors had the lowest influence. Furthermore, this paper identifies popular wine topics and keywords, in addition to exploring the sentiment in the sommeliers’ reviews to determine if those two factors also have an affect on a wine’s rating. Lastly, a kNN-based recommender system is created to suggest similar options based on one given wine. 
	
The purpose of this analysis is to help wine retailers strategically identify the most popular wines based on this study, which will presumably increase sales. Retailers can use the kNN recommender system to suggest new wines to customers based on other wines they prefer in order to make the best suggestions and increase both customer loyalty and customer lifetime value. As a result, this may lead to a decrease in turnover inventory as products will sell faster, allowing retailers to regularly purchase new wines and diversify their product selection.

## Steps

 1. Data Cleaning
 2. Data Analysis
 3. Topic Modeling
 4. Logistic Regression
 5. kNN Recommender System

## Requirements

**Python.** Python is an interpreted, high-level and general-purpose programming language. 

**Integrated Development Environment (IDE).** Any IDE that can be used to view, edit, and run Python code, such as:
- [Google Colab](https://colab.research.google.com/notebooks/intro.ipynb#recent=true)
- [Jupyter Notebook](https://jupyter.org/).

### Packages 
Install the following packages in Python prior to running the code.
```python
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import re
import string

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

from wordcloud import WordCloud, STOPWORDS
from textblob import TextBlob
import nltk
nltk.download('punkt')
nltk.download('stopwords')

import spacy 

import gensim
from gensim import corpora

import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
%matplotlib inline
```

If using Google Colab, import drive.mount('/content/drive') and follow instructions in the output to authorize access to Google Drive in order to obtain directories.


## Launch
Download the Python File *Wine_Enthusiast.ipynb* and open it in the IDE. Download and import the dataset *wine.csv*. 

Download and unzip the *wine.csv* file.

Change the file path to the directory where the data file is located.

## Authors

[Silvia Ji](https://www.linkedin.com/in/silviaji/) - [GitHub](github.com/jisilvia)

Joshua Rotuna - [GitHub](https://github.com/joshrotuna)

## License
This project is licensed under the [MIT](https://choosealicense.com/licenses/mit/) License.

## Acknowledgements


The Dataset used was provided by Wine Enthusiast.
