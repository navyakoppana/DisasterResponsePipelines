# Disaster Response Pipeline Project
## Table of Contents
1. Introduction to the project
2. List of libraries used
3. List of files in repository
4. Instructions
5. Summary of the project
6. Conclusions

### Introduction to the project
The data set is provided by 'Figure Eight' which contains real messages that were sent during disaster events. 
In this project, we have to create ETL and ML pipeline to load and clean the datasets,then train and tune the model using GridSearchCV to categorize these events so that you can send the messages to an appropriate disaster relief agency.

### List of Libraries used
a. Pandas
b. Numpy
c. re
d. sqlalchemy
e. json
f. Plotly
g. FLask
h. Scikit-learn
i. NLTK
j. joblib
k. pickle

### List of files in the repository
There are three folders:
1) App: It contains templates and run.py file which uses Flask app to run web application.
2) Data:It contains csv files,database file and process_data.py file containing ETL pipeline.
3) Models:It has train_classifier.py file containing ML pipeline and pickle file.
There are ETL and ML ipynb files used in the Udacity project workspace.
There is README file which describes the libraries used, the files in repository,the instructions followed to carry out the project and summary(analysis) of the project.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage

### Summary of the project

Step-1:Build ETL pipeline.
a. Load the Categories & Messages csv.
b. Clean the categories file as required to carry out analysis.
c. Merge Data from the two csv files (messages.csv & categories.csv)
d. Remove duplicates.
e. Create SQL database DisasterResponse.db for the merged data sets.

Step-2:Text Preprocessing
a. Tokenize & lemmatize text

Step-3:Build Machine Learning Pipeline
a. Build Pipeline with countvectorizer and tfidtransformer, multioutput classifier with random forest classifier.
b. Train Pipeline (with Train/Test Split)
c. Print classification reports and accuracy scores
d. Improve Model using GridSearchCV and repeat steps b-c.
e. Export Model as .pkl File

Step-4: Web application using Flask app
a. Using Plotly in run.py build the graphs with the required parameters.
b. Following the Instructions from above, load the ETL pipeline to a database and  train the ML pipeline.
c. After running the run.py file the plotly visualisations can be viewed through Flask app and we can classify any message that users would enter on the web page.

#### Conclusions
From the plotly graphs,we can conclude that:
1. The messages related to aid have the highest count.
2. The order of messages count based on genres:news>direct>social

