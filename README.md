# DisasterResponsePipeline

### Objective:
The goal of this project is to build an API that classifies the messages received during the disaster into various categories. The actual messages received by Figure8 have been analysed for this project

### Project Workflow:

In this project, firstly an ETL pipleline has been built. This pipeline loads the messages.csv and categories.csv datasets, merges and cleans the dataset, and finally stores the data in sql database.

Secondly a machine learning pipeline has been built. This pipeline consists of text processing and machine learning. GridSearch cross validation has also been used. Adaboost multiclass classifier has been used for the training the model

### File Descriptions:

The files are organised as follows:

workspace
    - data
        -disaster_messages.csv: data file
        -disaster_categories.csv : data file
        -process_data.py : ETL pipeline script        
    - models
        -train_classifier: ML pipeline script
    - app
        -run.py: script for app visulaization
        -templates: html scripts to display webpage
            -master.html
            -go.html
    
### Instructions:

1. Run the following commands in the project's root directory to set up the database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run the web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


