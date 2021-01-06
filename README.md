# Springboard Data Science Immersive

![Collages](https://github.com/dimmani1835/Data-Science---Springboard/blob/main/images/collages.jpg)

This repository will house all code, data, and files related to my work in the Springboard Data Science Immersive program. The following acts as a table of contents for the whole repository with links to the respective work cited.

-----------------------------------------------------------------------------

# Capstone Projects (Full Scale Data Science Projects)

## [Building A Spam Filter With Natural Language Processing](https://github.com/dimmani1835/Data-Science---Springboard/tree/main/SMS%20Spam%20Filter)

![Spam Filtering](https://github.com/dimmani1835/Data-Science---Springboard/blob/main/images/Sms%20filter.PNG)

### Key points:
- Python tools: Pandas, Json, Nltk, Matplotlib, Scikit-Learn, Spacy, Statsmodel.
- Performed spam classification for hypothetical a hypothetical telecom, achieving a model with 99.4% accuracy score and 0.997 AUC (Area Under the Curve) Score.
- Created a complete end-to-end NLP pipeline to process sms messages at throughput of +100 sms/s.
- Converted data to bag of word features and modeled the training data to several algorithms, including Logistic regression, Naive Bayes, SVC and XGBoost.
- Regularized best algorithm architecture with F-Beta score to ensure alignment with business objectives.
- Deployed [Spam Filter](http://spamfilter-env.eba-5zbpxzwv.us-east-2.elasticbeanstalk.com/) onto AWS instance with Dockerization and Flask so that anyone could filter Spams.

## [Sentiment Analysis with Neural Network techniques](https://github.com/dimmani1835/Data-Science---Springboard/tree/main/Drug%20Sentiment%20Review)

![Drug Sentiment Classification](https://github.com/dimmani1835/Data-Science---Springboard/blob/main/images/Sentiment_Analysis_drug.PNG)

### Key Points:
- Python tools: Pandas, Seaborn, Scikit-Learn, Spacy, Gensim, Sci-Py, Tensorflow/Keras.
- Performed sentiment analysis on a highly realistic dataset provided by Analytics Vidha with over 100 drugs for a hypothetical Canadian pharmaceutical, achieving a model with 0.5 F1 macro score with Grid-Search for Hyper-parameter Tuning and Cross Validation.
- Aligned the best algorithm with business objectives by creating trade-off gradient to identify scenarios in which automation of the Sentiment Analysis task is profitable, potentially with a saving of +$400 per 1000 reviews.
- Ran dimensionality reduction algorithms such as TSNE and LDA (Latent Dirichlet Allocation) to explore language dataset structure.
- Modeled the training data to several algorithms, including Word embedding algorithms (with Skip-Gram, Matrix Factorization and Transfer Learning) and Recurrent Neural Network with LSTM (Long Short Term Memory) component.

-----------------------------------------------------------------------------

# Mini Projects (Small Data Science Projects showcasing specific skills)

## [SQL - Time Series Analysis](https://github.com/dimmani1835/Data-Science---Springboard/blob/main/Mini-Projects%20and%20Exercises/SQL/country_clubs_analytics.ipynbt)
### Description
This is a SQL case study on Country Club membership as proposed from Mode Analytics at https://modeanalytics.com/. The Jupyter notebook in this repository is a cleaned up verison of the original case study which contains all original SQL queries, and can be found here: https://modeanalytics.com/mooseburger/reports/14cbbb5670b8
### Key Skills:
- SQL Query
- SQL Plotting and Graphing
- Advanced SQL Queries (Join, Merge, Subqueries)

## [JSON wrangling](https://github.com/dimmani1835/Data-Science---Springboard/blob/main/Mini-Projects%20and%20Exercises/Data%20Wrangling/API_import_and_wrangling.ipynb)
### Description:
Importing stock data from Quandl and conducting data analysis and exploration of stock data using native Python structures.

### Key skills
- API import, request
- Data extraction
- native Python structures

## [Data API Import](https://github.com/zachnguyen/Data-Science---Springboard/blob/main/Mini-Projects%20and%20Exercises/Data%20Wrangling/json_wrangling_worldbank.ipynb)
### Description:
Importing country data from world bank and conducting data analysis and exploration of Json structures.

### Key skills
- json manipulation
- missing data imputation 

## [Clustering Methods K-Nearest Neighbors and PCA](https://github.com/dimmani1835/Data-Science---Springboard/blob/main/Mini-Projects%20and%20Exercises/Unsupervised%20Learning/Mini_Project_Clustering_Customer_Segmentation.ipynb)
### Description
A mini project on customer segmentation and being able to identify different types of customers and then figure out ways to find more of those individuals so you can get more customers! The data comes from John Foreman's book Data Smart. The dataset contains both information on marketing newsletters/e-mail campaigns (e-mail offers sent) and transaction level data from customers (which offer customers responded to and what they bought).
### Key Skills
- K-Means
- PCA - Principle Component Analysis
- Elbow Sum of Squares Method

## [Exploratory Data Analysis with Frequentist, Bootstrap and Baysian statistical Methods](https://github.com/dimmani1835/Data-Science---Springboard/tree/main/Mini-Projects%20and%20Exercises/Inferential%20Statistics)
### Description
Several EDA's performed on varying data categories using three statistical methods to answer several questions relating to insurance claim in a hospital. 
### Key Skills
- Central Limit Theorem
- Statistical Analysis
- Data Visualization
- z-test
- t-test
- Margin of Error (MOE)
- Chi-Squared Test
- Bootstrap Statistics

## [Linear Regression](https://github.com/dimmani1835/Data-Science---Springboard/blob/main/Mini-Projects%20and%20Exercises/Linear%20Regression/Mini_Project_Linear_Regression_Boston_HousePrice.ipynb)
### Description
Applied Linear Regression onto Boston Housing Dataset to predict price.
### Key Skills:
- Regression
- Model evaluation (Rquared)
- Feature selection
- Q-Q Plot, residuals and outliers

## [Logistic Regression](https://github.com/dimmani1835/Data-Science---Springboard/blob/main/Mini-Projects%20and%20Exercises/Logistic%20Regression/Mini_Project_Logistic_Regression.ipynb)
### Description
Classify gender based on height using Logistic Regression. 
### Key skills:
- Classification
- Model evaluation (F-score, ROC)
- Log Probability, Optimization

## [Naive Bayes](https://github.com/dimmani1835/Data-Science---Springboard/blob/main/Mini-Projects%20and%20Exercises/Naive%20Bayes/Mini_Project_Naive_Bayes.ipynb)
### Description
Using Naive Bayes for basic text classification of "Rotten" movie reviews.
### Key skills:
- Text Classification
- Bag of Word, TF-IDF feature engineering
- Cross Validation
- Grid Search and Hyper-Parameter tuning
- Latent Dirichle Allocation

## [PySpark](https://github.com/dimmani1835/Data-Science---Springboard/blob/main/Mini-Projects%20and%20Exercises/PySpark/Spark%20DF%2C%20SQL%2C%20ML%20Exercise.ipynb)
### Description
Using MapReduce with Pyspark with several exercises utlitizing MapReduce Pyspark (RDD) with a touch of MLlib.
### Key Skills
- Pyspark
- RDD
- Spark Dataframes

## [Data Science Chalenge - Ultimate Inc.](https://github.com/dimmani1835/Data-Science---Springboard/tree/main/Mini-Projects%20and%20Exercises/Data%20Challenges/Ultimate%20Challenge)
### Description
Ultimate Challenge - Building a model to help figure out customer retention based on a variety of features including login information in the "logins.csv" dataset.
### Key Skills
- EDA
- Time series analysis
- Experimental design
- Feature engineering

## [Data Science Challenge - Relax Inc.](https://github.com/dimmani1835/Data-Science---Springboard/tree/main/Mini-Projects%20and%20Exercises/Data%20Challenges/Relax%20Challenges)
### Description
Relax Challenge - Defining an "adopted user" as a user who has logged into a product on three separate days in at least one seven-day period, identify which factors predict future user adoption given two datasets
- A user table ("takehome_users") with data on 12,000 users who signed up for the product in the last two years
- A usage summary table ("takehome_user_engagement") that has a row for each day that a user logged into the product.
### Key skills
- datetime wrangling
- full-stack data science
