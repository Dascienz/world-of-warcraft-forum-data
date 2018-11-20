#!/usr/bin/env python3
import numpy as np
import pandas as pd
import MySQLdb as sql

"""
The WoW Forum Data is stored in a MySQL database.
"""

#JSON credentials
login = pd.read_json(os.path.join(os.getcwd(),"db_credentials.json"), typ="series").to_dict()

def connect():
    #db connection
    conn = mysql.connect(host=login["host"],
                     user=login["user"],
                     passwd=login["passwd"],
                     db=login["db"])           
    cur = conn.cursor()
    return conn, cur

def readSamples(cols):
    conn, cur = connect()
    QUERY = "SELECT %s FROM forum_data ORDER BY RAND() LIMIT 100000" % (",".join(cols))
    data = pd.read_sql(QUERY, con=conn)
    return data

"""
Let's see if we can make an ambitious model for predicting 
in-game race based solely on how a player types on the forums. 
For this task, we'll query 100,000 random posts from our MySQL database.
"""

data = readSamples(['lvlRaceClass','postText'])

Factions = {r'.*Human.*': 'Alliance',
            r'.*Dwarf.*': 'Alliance',
            r'.*Gnome.*': 'Alliance',
            r'.*Night Elf.*': 'Alliance',
            r'.*Draenei.*': 'Alliance',
            r'.*Worgen.*': 'Alliance',
            r'.*Orc.*': 'Horde',
            r'.*Troll.*': 'Horde',
            r'.*Undead.*': 'Horde',
            r'.*Tauren.*': 'Horde',
            r'.*Goblin.*': 'Horde',
            r'.*Blood Elf.*': 'Horde',
            r'.*Pandaren.*': 'Pandaren',
            r'.*NaN.*': 'Staff'}

Races = {r'.*Human.*': 'Human',
         r'.*Dwarf.*': 'Dwarf',
         r'.*Gnome.*': 'Gnome',
         r'.*Night Elf.*': 'Night Elf',
         r'.*Draenei.*': 'Draenei',
         r'.*Worgen.*': 'Worgen',
         r'.*Orc.*': 'Orc',
         r'.*Troll.*': 'Troll',
         r'.*Undead.*': 'Undead',
         r'.*Tauren.*': 'Tauren',
         r'.*Goblin.*': 'Goblin',
         r'.*Blood Elf.*': 'Blood Elf',
         r'.*Pandaren.*': 'Pandaren',
         r'.*NaN.*': 'Staff'}

data['faction'] = data['lvlRaceClass']
data['race'] = data['lvlRaceClass']
data['faction'] = data['faction'].replace(Factions, regex=True)
data['race'] = data['race'].replace(Races, regex=True)


"""
We can vectorize posts and try to predict in-game race from post text using a simple SVM model.
"""

import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

def my_tokenizer(words):
    lem = WordNetLemmatizer()
    stopset = stopwords.words('english')
    return [lem.lemmatize(word,'n').lower() for
            word in nltk.word_tokenize(words) if word.lower() not in stopset and len(word.lower()) >= 3 and len(word.lower()) <= 20]

#Set our X (features) and Y (labels) columns.
posts_tokenized = data['postText'][(~data['race'].str.contains("Staff"))].apply(lambda s: my_tokenizer(s))
races = data['race'][(~data['race'].str.contains("Staff"))]

df = pd.DataFrame({'race': races, 'text': posts_tokenized})
df = df[df['text'].str.len() != 0].reset_index(drop=True)

import itertools
def vocabSet(lst):
    total = list(itertools.chain.from_iterable(lst))
    return list(set(total))

def fitVocabSet(text, vocabulary):
    arrays = []
    for lst in text:
        array = np.zeros((len(vocabulary)))
        for word in lst:
            array[vocabulary.index(word)] = 1.
        arrays.append(array.astype('float32'))
    return arrays

import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def trainNetwork():
    """Set features and labels."""
    vocab = vocabSet(df['text'])
    X = df['text']
    Y = pd.get_dummies(df['race']).astype('float32')
    f_train, f_test, l_train, l_test = train_test_split(X, Y, test_size = 0.25)
    
    """Construct the network/graph."""
    inputs = tf.placeholder(tf.float32, [None, len(vocab)])
    weights = tf.Variable(tf.truncated_normal([len(vocab), len(races.unique())]))
    biases = tf.Variable(tf.zeros([len(races.unique())]))
    out = tf.nn.softmax(tf.matmul(inputs, weights) + biases)
    labels = tf.placeholder(tf.float32, [None, len(races.unique())])
    
    cross_entropy = -tf.reduce_sum(labels*tf.log(out))
    train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)
    
    evaluate = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out,1), tf.argmax(labels,1)), tf.float32))
    
    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer())

        """Perform batch training on one-hot-encoded labels and vectorized posts."""   
        for step in range(100000):
            f_train, l_train = shuffle(f_train, l_train)
            f_test, l_test = shuffle(f_test, l_test)
            
            f_train_batch = fitVocabSet(f_train.values[0:10], vocab)
            f_test_batch = fitVocabSet(f_test.values[0:10], vocab)
            l_train_batch = l_train.values[0:10]
            l_test_batch = l_test.values[0:10]
            
            TRAIN_FEED = {inputs: f_train_batch, labels: l_train_batch}
            TEST_FEED = {inputs: f_test_batch, labels: l_test_batch}

            sess.run(train_step, feed_dict=TRAIN_FEED)
            if(step%500 == 0):
                print("\nStep: ", step)
                print("Accuracy: ", sess.run(evaluate, feed_dict=TEST_FEED))