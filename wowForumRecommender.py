#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 15:17:32 2017

@author: Dascienz
"""

"""
Recommendation engine for potential amigos based on level, achievements, and faction.
We don't want to be racist nor do we want to pair the same classes up since different classes
have specific roles which complement one another.
"""

import numpy as np
import pandas as pd
import MySQLdb as sql
import matplotlib.pyplot as plt

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

def readSample(cols):
    conn, cur = connect()
    QUERY = """
            SELECT %s FROM forum_data AS r1 
            JOIN (SELECT (RAND() * (SELECT MAX(id) FROM forum_data)) AS id) AS r2 
            WHERE r1.id >= r2.id 
            ORDER BY r1.id ASC
            LIMIT 1000000;
            """ % (",".join(cols))
    data = pd.read_sql(QUERY,con=conn)
    return data

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

Levels = {r".*NaN.*": np.nan,
          r"[a-zA-z ]": ""}

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

class Recommender():

    def __init__(self):
        self.item_arrays = item_arrays
        self.length = len(item_arrays)
    
    def cosine_distance(self,x,y):
        cos = np.dot(x,y) / (np.sqrt(np.dot(x,x) * np.dot(y,y)) + 1e-9)
        return 1. - cos

    def cosine_similarity(self,item,item_arrays):
        distances = np.zeros(self.length)
        for idx in range(self.length):
            distances[idx] = self.cosine_distance(item,self.item_arrays[idx])
        return distances
            
    def recommend(self,idx,matches):
        player = self.item_arrays[idx]
        print("\nPlayer: "+str(df['lvlRaceClass'][idx])+str(" ")+str(df['achievements'][idx]))
        distances = self.cosine_similarity(player,self.item_arrays)
        ids = distances.argsort()[1:matches]
        print("\nRecommendations: ")
        for i in range(len(ids)):
            print(str(i+1) + ". " + str(df['lvlRaceClass'][ids[i]])+str(" ")+str(df['achievements'][ids[i]]))