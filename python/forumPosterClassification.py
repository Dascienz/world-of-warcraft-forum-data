#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 15:55:05 2017

@author: Dascienz
"""

import numpy as np
import pandas as pd
import MySQLdb as sql

def connect():
    conn = sql.connect(host="localhost",
                       user="root",
                       passwd="Byleth88$",
                       db="wow_forums")           
    cur = conn.cursor()
    return conn, cur


def readSample(cols):
    conn, cur = connect()
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

"""Let's see if we can predict whether someone will return to the forums based on the
content of their first post content."""

model = readSample(['lvlRaceClass','postCount','postText'])

"""We need to set our labels for one-time posters vs. return posters."""
model['posCount'] = model['postCount'].replace({r'[a-zA-z]+': np.nan}, regex=True)
model = model.dropna().reset_index(drop=True)
model['postCount'] = model['postCount'].astype('int')
model['poster'] = np.where(model['postCount'] == 1, 'One-off Poster', 'Return Poster')

"""Let's extract level data from the players to add as a feature."""
Levels = {r".*NaN.*": np.nan,
          r"[a-zA-z ]": ""}
model['level'] = model['lvlRaceClass'].replace(Levels, regex=True)
model = model.dropna().reset_index(drop=True)
model['level'] = model['level'].astype('int')
model['level'] = (model['level']-model['level'].min())/(model['level'].max()-model['level'].min())

from sklearn.utils import shuffle
model = shuffle(model)

#Vectorize post text.
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

def my_tokenizer(words):
    lem = WordNetLemmatizer()
    return [lem.lemmatize(word,'n').lower() for
            word in nltk.word_tokenize(words) if len(word.lower()) >= 3]

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(ngram_range=(1,3), stop_words='english', tokenizer=my_tokenizer)
vector_matrix = vectorizer.fit_transform(model['postText'])

import sklearn
from scipy.sparse import hstack
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

features = hstack((vector_matrix, np.array(model['level'])[:, None]))
labels = model['poster']

f_train, f_test, l_train, l_test = train_test_split(features, labels, test_size = 0.3)

for score in ['precision','recall']:
    
    parameters = [{'class_weight':['balanced'],'C':[65.0, 75.0, 85.0]}]
    
    print("### Tuning hyper-parameters for %s ###" % score)
    print()

    clf = GridSearchCV(LogisticRegression(), parameters, n_jobs=4, 
                       cv = 5, scoring='%s_macro' % score)
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