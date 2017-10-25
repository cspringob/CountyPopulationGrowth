from math import radians, cos, sin, asin, sqrt
import pandas as pd
import numpy as np
from sklearn import cluster
import random
import matplotlib.pyplot as plt

def select_years(df, yearlist):
    #Return a dataframe that only includes rows from the relevant years in "yearlist":
    dfout = df[df['YEAR'].isin(yearlist)].copy()
    return dfout

def calc_distance(df, geo1, geo2):
    #Calculates distance between two points on the globe.  This is essentially copied from here: https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points
    #Returns distance in kilometers
    lat1 = df[df['GEOID'] == geo1]['INTPTLAT'].values[0]
    lon1 = df[df['GEOID'] == geo1]['INTPTLONG'].values[0]
    lat2 = df[df['GEOID'] == geo2]['INTPTLAT'].values[0]
    lon2 = df[df['GEOID'] == geo2]['INTPTLONG'].values[0]
    #print(lat1, lon1, lat2, lon2)
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlat = lat1 - lat2
    dlon = lon1 - lon2
    #print(lat1, lat2, dlat)
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c*r

def make_correlation_function(df, num):
    #Calculates spatial autocorrelation function on population growth from dataframe df.  I'm only using a random slice of the data here, since the program is really slow:
    df2 = df.reset_index(drop = True)
    mymean = np.mean(df2['POP_GROWTH_P1'])
    myvar = np.std(df2['POP_GROWTH_P1'])
    diffarr = np.zeros(50)
    countarr = np.zeros(50)
    for i in range(0, num):
        r = random.randint(0, len(df2))
        for j in range(0, len(df2)):
            dist = calc_distance(df2, df2['GEOID'][r], df2['GEOID'][j])
            if((dist < 1000.0) and (r != j)):
                mybin = int(dist / 20.0)
                diffarr[mybin] = diffarr[mybin] + (df2['POP_GROWTH_P1'][r] - mymean) * (df2['POP_GROWTH_P1'][j] - mymean)
                countarr[mybin] = countarr[mybin] + 1.0
        print(r, mybin, countarr[mybin])
    corrarr = (1.0 / countarr) * (diffarr / (myvar * myvar))
    distarr = np.zeros(50)
    for i in range(0,50):
        distarr[i] = 20.0 * i + 10.0
    return distarr, corrarr, countarr


def kmeans(df, num_c):
    # Creates clones of counties with large populations, and then runs kmeans on the combined dataset.
    df2 = df.reset_index(drop = True)
    #Keeping a separate copy "df3" ,so as not to get confused in the loop about the length of the dataframe.
    df3 = df2.copy()
    for i in range(0, len(df2)):
        if(df2['0'][i] > 400000):
            repeat = int(df2['0'][i] / 200000) - 1
            for j in range(0,repeat):
                df3 = df3.append(df3.loc[i], ignore_index = True)
        print(i)
    df4 = df3[['INTPTLAT', 'INTPTLONG']].copy()
    k_means = cluster.KMeans(n_clusters=num_c)
    k_means.fit(df4)
    df2['cluster'] = k_means.labels_[:len(df2)]
    return df2

def assign_teams(df):
    #Add a column to the dataframe that tells you which split that cluster is in.  Split == 0 will be the test split.
    ranlist = []
    teamarr = np.zeros(102)
    for i in range(0,6):
        for j in range(0,17):
            ranlist.append(i)
    for i in range(0,102):
        myran = random.randint(0,len(ranlist)-1)
        teamarr[i] = ranlist[myran]
        ranlist.pop(myran)
    df['split'] = teamarr[df['cluster']].astype(int)
    return df

if __name__ == '__main__':
    #Read in data, and create the features:
    df = pd.read_csv('../DATA/year_rawfeature.csv')
    dfy = pd.read_csv('../DATA/y1_df.csv')

    X2008 = select_years(df, [2008])
    y2008 = select_years(df, [2008])
    #Calculate the spatial correlation function:
    """dists, corrs, counts = make_correlation_function(X2008, 100)
    d = {'distance[km]': dists, 'correlation': corrs, 'counts': counts}
    corrs_df = pd.DataFrame(data = d)
    corrs_df.to_csv('../DATA/corr_func.csv')"""
    clusterdf = kmeans(X2008, 102)
    #clusterdf.to_csv('../DATA/clusters3.csv')
    teams_df = assign_teams(clusterdf)
    #Let's just double check that we haven't missed any counties:
    print(set(df['GEOID']) - set(teams_df['GEOID']))
    print(set(teams_df['GEOID']) - set(df['GEOID']))
    #OK, we have them all. Let's save this to a file
    teams_df.to_csv('../DATA/splits.csv')
    #Now, let's add these assignments back into the parent dataframe:
    teamcopy = teams_df[['GEOID', 'cluster', 'split']].copy()
    allteams_df = pd.merge(df, teamcopy, how='left', left_on='GEOID', right_on='GEOID')
    #And save this to a file:
    allteams_df.to_csv('../DATA/allsplits.csv')
