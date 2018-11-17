# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 08:32:07 2017

@author: David
"""

import time
import datetime
import numpy as np
import pandas as pd
import MySQLdb as sql
import matplotlib.pyplot as plt
from matplotlib import cm as cm

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

def readData(cols):
    conn, cur = connect()
    QUERY = "SELECT %s FROM forum_data" % (",".join(cols))
    data = pd.read_sql(QUERY, con=conn)
    return data

def dateConversion(series):
    dates = {date: pd.to_datetime(date).date() for date in series.unique()}
    return series.map(dates)

def timestampConversion(series):
    timestamps = {date: time.mktime(date.timetuple()) for date in series.unique()}
    return series.map(timestamps)

"""WoW Forum Activity over the last 7 years. Declining?"""

data = readData(['datetime'])

data['datetime'] = dateConversion(data['datetime'])

dates = data['datetime']
counts = dates.value_counts()
dt_dict = {date: time.mktime(date.timetuple()) for date in counts.index}
timestamps = dates.map(dt_dict)

def timeSeries(dates=timestamps, counts=timestamps.value_counts()):
    """Forum activity over the last 7 years."""
    cmap = plt.cm.get_cmap('winter')
    Y,X = np.histogram(timestamps,len(counts),normed=1.0)
    y_span = Y.max()-Y.min()
    colors = [cmap(((y-Y.min())/y_span)) for y in Y]
    y_max = counts.sum()
    
    plt.figure(figsize=(20,5))
    plt.title('US WoW Forums: Last 7 Years')
    plt.ylabel('Forum Activity')
    plt.bar(X[:-1],Y*y_max, width=((X.max()-X.min())/(len(X))), color=colors, align='center',linewidth=0)
    locs = np.arange(X.min(),X.max()+1, ((X.max()-X.min())/7))
    labs = [datetime.date.fromtimestamp(x) for x in locs]
    
    cata_date = time.mktime(datetime.date(2010,12,7).timetuple())
    plt.axvline(x=cata_date,color='red',linestyle='dashed')
    plt.text(cata_date, 0.8,'Cataclysm', color='red', rotation=45)
    
    mists_date = time.mktime(datetime.date(2012,9,25).timetuple())
    plt.axvline(x=mists_date,color='red',linestyle='dashed')
    plt.text(mists_date, 0.8,'Mists of Pandaria', color='red', rotation=45)
    
    warlords_date = time.mktime(datetime.date(2014,11,13).timetuple())
    plt.axvline(x=warlords_date,color='red',linestyle='dashed')
    plt.text(warlords_date, 0.8,'Warlords of Draenor', color='red', rotation=45)
    
    legion_date = time.mktime(datetime.date(2016,8,30).timetuple())
    plt.axvline(x=legion_date,color='red',linestyle='dashed')
    plt.text(legion_date, 0.8,'Legion', color='red', rotation=45)
    
    plt.xticks(locs, labs, ha='right',rotation=45)
    plt.axis('tight')
    plt.xlim([X.min(), X.max()])
    plt.tight_layout()
    plt.savefig('WoW_Forum_Activity.png', format='png',dpi=300)
    plt.show()
    
"""Have forum activities been declining? Unfortunately yes!"""

def yearPosts(a,b,counts):
    counts = counts
    idx = counts.index
    total = counts[(idx >= datetime.date(a,1,1)) & (idx <= datetime.date(b,1,1))].sum()
    return total

def yearActivity():
    """Total Year Activity"""
    plt.figure()
    plt.title('Forum Activity by Year')
    y = [yearPosts(2000+x,2000+x+1,counts) for x in range(11,18)]
    x = np.arange(0,7,1)
    labels = [str(2000+x) for x in range(11,18)]
    plt.xticks(x,labels)
    plt.bar(x,y, color='green',alpha=0.75)
    plt.ylabel('Posts')
    plt.xlabel('Year')
    plt.savefig('WoW_Forum_Activity_By_Year.png', format='png',dpi=300)
    plt.show()

"""Participation has been declining, but is this could also be due to improvements
elsewhere such as with the chat features on blizzard's battle.net portal or even
other third-party software such as discord. Which players, specifically are sticking around?"""

data = readData(['name','postCount','datetime'])

data['datetime'] = dateConversion(data['datetime'])
data['timestamps'] = timestampConversion(data['datetime'])

data['firstPost'] = data.groupby('name')['timestamps'].transform('min')
data['lastPost'] = data.groupby('name')['timestamps'].transform('max')
data['days'] = ((data['lastPost']-data['firstPost'])/(60*60*24)).replace({0.0:1.0})

expansions = ['Cataclysm', 'Mists of Pandaria', 'Warlords of Draenor', 'Legion']
cata_date = time.mktime(datetime.date(2010,12,7).timetuple())
mists_date = time.mktime(datetime.date(2012,9,25).timetuple())
warlords_date = time.mktime(datetime.date(2014,11,13).timetuple())
legion_date = time.mktime(datetime.date(2016,8,30).timetuple())
bins = [cata_date, mists_date, warlords_date, legion_date, data['timestamps'].max()]
data['expansion'] = pd.cut(data['timestamps'], bins, labels=expansions)

df = data.drop_duplicates().reset_index(drop=True)
def postByExpansion():
    """Individual player activity."""
    plt.figure()
    plt.title('WoW Forum Posts By Expansion')
    y = df['expansion'].value_counts().values
    x = np.arange(1,5,1)
    labels = ['Cataclysm', 'Mists of Pandaria', 'Warlords of Draenor', 'Legion']
    plt.bar(x,y, color='blue', alpha = 0.5)
    plt.xticks(x,labels, rotation=45)
    plt.ylabel('Player Posts')
    plt.axis('tight')
    plt.tight_layout(pad=0)
    plt.savefig('WoW_Forum_Activity_By_Expansion.png', format='png',dpi=300)
    plt.show()

def durations():
    """Average number of days spent active on forums per player."""
    plt.figure()
    x = df['days']
    plt.title('Average Duration of Forum Participation (Bins = 100)')
    plt.hist(x, normed=True, bins=100, color='red')
    plt.axis('tight')
    plt.ylabel('Duration Probability')
    plt.xlabel('Days')
    plt.xlim([x.min()-50,x.max()+50])
    plt.tight_layout()
    plt.savefig('WoW_Forum_Durations.png',format = 'png', dpi = 300)
    plt.show()