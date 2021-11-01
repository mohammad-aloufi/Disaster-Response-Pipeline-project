# Disaster Response Pipeline Project

# Project description

[Figure Eight Inc](https://en.wikipedia.org/wiki/Figure_Eight_Inc.) has provided a pre labeled dataset that contains tweets and messages that talk about real life disasters. I used this dataset to train a machine learning model that can classify a message into 36 categories and hopefully helping disaster response organizations  saving a lot of time.

## More about this project

From a technical perspective, this project contains 3 parts:

1. ETL pipeline.
2. Machine learning pipeline.
3. A flask app.

### ETL pipeline

ETL stands for extract, transform, load. At first, we have 2 datasets, disaster_messages.csv and disaster_categories.csv.
In the data folder, we have a script called process_data.py. and here is where the ETL pipeline is done. We loaded the datasets, joined them together, did the cleaning and data wrangling and then saved the new data into a database.

### Machine learning pipeline

In the models folder, we have the script train_classifier.py. This script loads the database and then builds an ml model, training it, evaluating it, and then saving it making everything ready for the next and last part.

### The flask app

The flask app uses both the database and the model to surve a frontend interface for the end user so they can actually use the models on unseen data. The user will type a message and then will hit the classify message button, the backend server will predict the message and send the response to the frontend enterface.

# Installation

You need python 3.8, and you need to install the following libraries:

pandas, numpy, sqlalchemy, pickle, nltk, sklearn, joblib, plotly, and flask

You can install all libraries with the following command while you're in the project folder:

pip install -r requirements.txt

## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/database.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/database.db model.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
