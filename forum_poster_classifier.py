#!/usr/bin/env python3
import numpy as np
import pandas as pd
import MySQLdb as sql

#working directory
wdir = os.path.dirname(__file__)

#json credentials
CREDENTIALS = pd.read_json(os.path.join(wdir,"db_credentials.json"), typ="series").to_dict()

##########################################################################################
# FUNCTIONS FOR CONNECTIONG TO MYSQL AND QUERYING DATA
##########################################################################################

def connection(login):
    """Function for establishing MySQL database connection.
    ----- Args:
            login: dict() containing hostname, username, password and database name.
    ----- Returns:
            MySQLdb.connect and MySQLdb.connect.cursor objects.
    """
    
    conn = mysql.connect(host=login["host"],
                     user=login["user"],
                     passwd=login["passwd"],
                     db=login["db"])           
    cur = conn.cursor()
    return conn, cur


def readSample(cols):
    """Specific function for querying rows from MySQL database.
    ----- Args:
            cols: list() of column names
    ----- Returns:
            data: pandas.DataFrame of shape (500000, len(cols))
    """
    
    conn, cur = connect(login=CREDENTIALS)
    QUERY_SINGLE_POSTS = """
                        SELECT %s FROM forum_data AS r1 JOIN
                        (SELECT (RAND() * (SELECT MAX(id) FROM forum_data)) AS id)
                        AS r2 WHERE (r1.id >= r2.id AND postCount = 1) ORDER BY r1.id ASC
                        LIMIT 500000;
                        """ % (",".join(cols))
    
    QUERY_MULTI_POSTS = """
                        SELECT %s FROM forum_data AS r1 JOIN
                        (SELECT (RAND() * (SELECT MAX(id) FROM forum_data)) AS id)
                        AS r2 WHERE (r1.id >= r2.id AND postCount > 1) ORDER BY r1.id ASC
                        LIMIT 500000;
                        """ % (",".join(cols))

    x = pd.read_sql(QUERY_SINGLE_POSTS, con=conn)
    y = pd.read_sql(QUERY_MULTI_POSTS, con=conn)
    data = pd.concat([x, y], axis=0)
    return data


##########################################################################################
# PREPARING DATA FOR MODELING
##########################################################################################


#Let's see if we can predict whether someone will return to the forums based on the
#content of their first post content."""

model = readSample(['lvlRaceClass','postCount','postText']) #data to model

#We need to set our labels for one-time posters vs. return posters
model['posCount'] = model['postCount'].replace({r'[a-zA-z]+': np.nan}, regex=True)
model = model.dropna().reset_index(drop=True) #drop NULL
model['postCount'] = model['postCount'].astype('int')
model['poster'] = np.where(model['postCount'] == 1, 'One-off Poster', 'Return Poster') #labels

#Let's extract level data from the players to add as a feature.
Levels = {r'.*NaN.*': np.nan, r'[a-zA-z ]': ''}
model['level'] = model['lvlRaceClass'].replace(Levels, regex=True)
model = model.dropna().reset_index(drop=True) #drop NULL
model['level'] = model['level'].astype('int')

def min_max_scaler(series):
    """Function for normalizing a pandas.Series using a 
    minimum-maximum scaling transform.
    ----- Args:
            series: pandas.Series
    ----- Returns:
            pandas.Series
    """
    
    return (series - series.min()) / (series.max() - series.min())

model['level'] = min_max_scaler(model['level']) #min-max scaling

from sklearn.utils import shuffle
model = shuffle(model) #shuffle data

#Vectorizing post text.
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

def my_tokenizer(words):
    """Function for tokenizing forum posts.
    ----- Args:
            words: (str) text
    ----- Returns:
            list() of lemmatized (str) tokens.
    """
    
    lem = WordNetLemmatizer()
    return [lem.lemmatize(word,'n').lower() for
            word in nltk.word_tokenize(words) if len(word.lower()) >= 3]

from sklearn.feature_extraction.text import TfidfVectorizer

#TFIDF vectorizer, form vector matrix from forum posts.
vectorizer = TfidfVectorizer(ngram_range=(1,3), stop_words='english', tokenizer=my_tokenizer)
vector_matrix = vectorizer.fit_transform(model['postText'])


##########################################################################################
# PREDICTING ONE-OFF VERSUS RETURN FORUM POSTERS
##########################################################################################


import sklearn
from scipy.sparse import hstack
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

features = hstack((vector_matrix, np.array(model['level'])[:, None]))
labels = model['poster']

f_train, f_test, l_train, l_test = train_test_split(features, labels, test_size = 0.3) #split training and test sets for CV

for score in ['precision','recall']:
    
    parameters = [{'class_weight':['balanced'],'C':[65.0, 75.0, 85.0]}]
    
    print("### Tuning hyper-parameters for %s ###" % score)
    print()

    clf = GridSearchCV(LogisticRegression(), parameters, n_jobs=4, 
                       cv=5, scoring='{}_macro'.format(score))
    clf.fit(f_train, l_train)

    print("Best parameter set found:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
        
    print()

    print("Classification report: ")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    l_true, l_pred = l_test, clf.predict(f_test)
    print(classification_report(l_true, l_pred))
    print()