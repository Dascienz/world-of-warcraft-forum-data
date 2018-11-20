#!/usr/bin/env python3
import numpy as np
import pandas as pd
import MySQLdb as sql
import matplotlib.pyplot as plt

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
            data: pandas.DataFrame of shape (1000000, len(cols))
    """
    
    conn, cur = connect(login=CREDENTIALS)
    QUERY = """
            SELECT {} FROM forum_data AS r1 
            JOIN (SELECT (RAND() * (SELECT MAX(id) FROM forum_data)) AS id) AS r2 
            WHERE r1.id >= r2.id 
            ORDER BY r1.id ASC
            LIMIT 1000000;
            """.format(",".join(cols))
    data = pd.read_sql(QUERY,con=conn)
    return data

##########################################################################################
# DATA PREPARATION
##########################################################################################


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

Classes = {r'.*Warrior.*': 'Warrior',
           r'.*Rogue.*': 'Rogue',
           r'.*Mage.*': 'Mage',
           r'.*Warlock.*': 'Warlock',
           r'.*Shaman.*': 'Shaman',
           r'.*Druid.*': 'Druid',
           r'.*Priest.*': 'Priest',
           r'.*Paladin.*': 'Paladin',
           r'.*Hunter.*': 'Hunter',
           r'.*Death Knight.*': 'Death Knight',
           r'.*Monk.*': 'Monk',
           r'.*Demon Hunter.*': 'Demon Hunter',
           r'.*NaN.*': 'Staff'}

Levels = {r".*NaN.*": np.nan, r"[a-zA-z ]": ""}

Achievements = {r".*NaN.*": np.nan}

data = readSample(['lvlRaceClass','achievements'])

data['faction'] = data['lvlRaceClass'].replace(Factions, regex=True)
data['level'] = data['lvlRaceClass'].replace(Levels, regex=True)
data['achievements'] = data['achievements'].replace(Achievements, regex=True)

df = data[['level','achievements','faction','lvlRaceClass']].copy()
df = df.dropna().reset_index(drop=True)
df['faction'] = pd.get_dummies(df['faction'])
df = df.drop_duplicates().reset_index(drop=True)
player_arrays = df[['level','achievements','faction']].as_matrix().astype('float')

##########################################################################################
# RECOMMENDATIONC CLASS AND METHODS
##########################################################################################

class Recommender():
    """Class for generating recommended matches using a 
    cosine distance-based algorithm."""

    def __init__(self):
        self.item_arrays = item_arrays
        self.length = len(item_arrays)
    
    def cosine_distance(self, x, y):
        """Function for calculating cosine distance
        between two arrays.
        ----- Args:
                x: np.array
                y: np.array
        ----- Returns:
                (float) cosine distance
        """
        
        epsilon = 1e-9
        a = np.dot(x,y) #numerator
        b = np.sqrt(np.dot(x,x) * np.dot(y,y)) + epsilon #denominator
        return 1. - (a / b)

    def cosine_similarity(self, iArray, nArrays):
        """Function for calculating cosine similarities
        between a given array and an array of arrays.
        ----- Args:
                x: np.array of shape (1,S)
                y: multi-dimensional array of shape (n,S)
        ----- Returns:
                list() of cosine distances
        """

        return [cosine_distance(iArray,x) for x in nArrays]
            
    def recommend(self, idx, matches):
        """Function for generating a list of recommendations.
        ----- Args:
                idx: (int) index
        ----- Returns:
                None
        """
        
        player = self.item_arrays[idx] #player
        
        print("\nPlayer: " + str(df['lvlRaceClass'][idx]) + str(" ") + str(df['achievements'][idx]))
        
        distances = self.cosine_similarity(iArray=player, nArrays=self.item_arrays) #list of cosine distances
        
        ids = distances.argsort()[1:matches] #list of sorted ids, minus zeroth element since this is the player array
        
        #print out recommendations
        n = len(ids)
        print("\nRecommendations: ")
        for i in range(n):
            print(str(i+1) + ". " + str(df['lvlRaceClass'][ids[i]]) + str(" ") + str(df['achievements'][ids[i]]))
