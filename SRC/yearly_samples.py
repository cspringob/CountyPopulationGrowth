import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats
from scipy.stats import norm
from math import radians, cos, sin, asin, sqrt

def feature_engineer(mydf):
    #This function adds the relevant features, and converts several of them into log units.
    df = mydf.copy()
    df = df.drop('Unnamed: 0', 1)
    #First, add the fractions for age cohorts:
    for i in range(0,18):
        df[''.join(['age_cohort', str(i + 1)])] = df[str(i + 1)].astype(float) / df[str(0)].astype(float)
    #Now some log versions of other parameters:
    df['logpop'] = np.log10(df[str(0)].astype(float))
    df['logland'] = np.log10(df['ALAND_SQMI'])
    #Need to set low values of water square mileage to a positive value first:
    df.loc[df['AWATER_SQMI'] < 0.01, 'AWATER_SQMI'] = 0.01
    df['logwater'] = np.log10(df['AWATER_SQMI'])
    df['fem_frac'] = df[str(200)].astype(float) / df[str(0)].astype(float)
    return df

def select_years(df, yearlist):
    #Return a dataframe that only includes rows from the relevant years in "yearlist":
    dfout = df[df['YEAR'].isin(yearlist)].copy()
    return dfout

def select_cols(df, collist):
    #Returns a dataframe with only the selected columns
    dfout = df[collist].copy()
    return dfout

def kill_outliers(X, y):
    #Identifies outliers, and removes them from the dataframes:
    outlierlist = []
    cols = ['POP_GROWTH_P1', 'POP_GROWTH_P2', 'POP_GROWTH_P3', 'POP_GROWTH_P4', 'POP_GROWTH_P5']
    for i in range(0, len(cols)):
        mymean = X[cols[i]].mean(axis = 0)
        mystd = X[cols[i]].std(axis = 0)
        for index in X[cols[i]].index:
            if(abs(X[cols[i]][index] - mymean) > (5.0 * mystd)):
                outlierlist.append(index)
    coly = 'POP_GROWTH_F1'
    mymean = y[coly].mean(axis = 0)
    mystd = y[coly].std(axis = 0)
    for index in y[coly].index:
        if(abs(y[coly][index] - mymean) > (5.0 * mystd)):
            outlierlist.append(index)
    setlist = set(outlierlist)
    Xout = X.loc[~X.index.isin(setlist)]
    yout = y.loc[~y.index.isin(setlist)]
    return Xout, yout, setlist

def find_outliers(X):
    #Same as kill outliers, except it only identifies the outliers and puts them in a list, rather than eliminate them.  The list is a list of tubles, which tells us which of the features it's an outlier on.
    outlierlist = []
    cols = ['POP_GROWTH_P1', 'POP_GROWTH_P2', 'POP_GROWTH_P3', 'POP_GROWTH_P4', 'POP_GROWTH_P5']
    for i in range(0, len(cols)):
        mymean = X[cols[i]].mean(axis = 0)
        mystd = X[cols[i]].std(axis = 0)
        for index in X[cols[i]].index:
            if(abs(X[cols[i]][index] - mymean) > (5.0 * mystd)):
                outlierlist.append((index, i))
    return outlierlist

def kill_more_outliers(X, y, cols):
    #Identifies outliers, and removes them from the dataframes.  This works the same as "kill_outliers", except that rather than using a fixed list of columns, it accepts user input.
    outlierlist = []
    for i in range(0, len(cols)):
        mymean = X[cols[i]].mean(axis = 0)
        mystd = X[cols[i]].std(axis = 0)
        for index in X[cols[i]].index:
            if(abs(X[cols[i]][index] - mymean) > (5.0 * mystd)):
                outlierlist.append(index)
    coly = 'POP_GROWTH_F1'
    mymean = y[coly].mean(axis = 0)
    mystd = y[coly].std(axis = 0)
    for index in y[coly].index:
        if(abs(y[coly][index] - mymean) > (5.0 * mystd)):
            outlierlist.append(index)
    setlist = set(outlierlist)
    Xout = X.loc[~X.index.isin(setlist)]
    yout = y.loc[~y.index.isin(setlist)]
    return Xout, yout, setlist

def engineer_clusters(df, outlist = [(0, 10)], sigma = 50, flag = True):
    #Reads in a dataframe that includes cluster assignments, and engineers some new cluster-specific features.  This should be run after feature_engineer has been run, and possibly after the years have been selected (that's optional).  But run it before you start eliminating columns with select_cols.
    #Columns to create: total population in the cluster, total land area in the cluster, 1 year population growth in the cluster, 2 year population growth in the cluster.  All of those in log units.
    #The first part is something that you must run through the first time you process the data.  So it only executes when flag = True:
    if(flag == True):
        #First, calculate last year's populations, and the year before:
        df2 = df.copy()
        df2['pop1'] = df2['0'] * (10.0 ** (-1.0 * df2['POP_GROWTH_P1']))
        df2['pop2'] = df2['0'] * (10.0 ** (-1.0 * df2['POP_GROWTH_P2']))
        groupdf = df2.groupby(['YEAR', 'cluster'], as_index=False).sum()
        groupdf['logpop_clu'] = np.log10(groupdf['0'].astype(float))
        groupdf['logland_clu'] = np.log10(groupdf['ALAND_SQMI'])
        groupdf['pop_growth_clu1'] = np.log10(groupdf['0'].astype(float) / groupdf['pop1'])
        groupdf['pop_growth_clu2'] = np.log10(groupdf['0'].astype(float) / groupdf['pop2'])
        #Now merge that back into the big table:
        mycols = ['YEAR', 'cluster', 'logpop_clu', 'logland_clu', 'pop_growth_clu1', 'pop_growth_clu2']
        grouptab = select_cols(groupdf, mycols)
        df3 = pd.merge(df2, grouptab, how='left', left_on=['YEAR', 'cluster'], right_on=['YEAR', 'cluster'])

        #Create category "group" for unique cluster in a given year:
        #sigma = 50.0
        df3['group'] = 1000 * df3['YEAR'] + df3['cluster']
    else:
        df3 = df.copy()

    #OK, now for the Gaussian smoothing:
    grouplist = df3['group'].unique()
    mycols = ['growthweight1', 'growthweight2', 'growthweight3', 'growthweight4', 'growthweight5', 'popweight', 'denseweight']
    listcols = []
    for item in mycols:
        listcols.append(''.join([item, str(sigma)]))
    for i in listcols:
        df3[i] = 0.0
    #print(grouplist)
    for i in range(0,len(grouplist)):
        for index, row in df3[df3['group'] == grouplist[i]].iterrows():
            #print(index, grouplist[i])
            weightsum = 0.0
            weightpopsum = 0.0
            growthsum1 = 0.0
            growthsum2 = 0.0
            growthsum3 = 0.0
            growthsum4 = 0.0
            growthsum5 = 0.0
            popsum = 0.0
            denssum = 0.0
            for index2, row2 in df3[df3['group'] == grouplist[i]].iterrows():
                mydist = calc_dist(row['INTPTLAT'], row['INTPTLONG'], row2['INTPTLAT'], row2['INTPTLONG'])
                weight = norm.pdf(mydist/float(sigma))
                weightsum = weightsum + weight
                weightpop = weight * row2['0']
                weightpopsum = weightpopsum + weightpop
                if((index == index2) or ((index2, 0) not in outlist)):
                    growthsum1 = growthsum1 + row2['POP_GROWTH_P1'] * weightpop
                else:
                    growthsum1 = growthsum1 + row['POP_GROWTH_P1'] * weightpop
                if((index == index2) or ((index2, 1) not in outlist)):
                    growthsum2 = growthsum2 + row2['POP_GROWTH_P2'] * weightpop
                else:
                    growthsum2 = growthsum2 + row['POP_GROWTH_P2'] * weightpop
                if((index == index2) or ((index2, 2) not in outlist)):
                    growthsum3 = growthsum3 + row2['POP_GROWTH_P3'] * weightpop
                else:
                    growthsum3 = growthsum3 + row['POP_GROWTH_P3'] * weightpop
                if((index == index2) or ((index2, 3) not in outlist)):
                    growthsum4 = growthsum4 + row2['POP_GROWTH_P4'] * weightpop
                else:
                    growthsum4 = growthsum4 + row['POP_GROWTH_P4'] * weightpop
                if((index == index2) or ((index2, 4) not in outlist)):
                    growthsum5 = growthsum5 + row2['POP_GROWTH_P5'] * weightpop
                else:
                    growthsum5 = growthsum5 + row['POP_GROWTH_P5'] * weightpop
                #popsum = popsum + row2['logpop'] * weight
                popsum = popsum + row2['0'] * weight
                #denssum = denssum + (row2['logpop'] - row2['logland']) * weight
                denssum = denssum + (row2['0'] / row2['ALAND_SQMI']) * weight
            listvals = [growthsum1, growthsum2, growthsum3, growthsum4, growthsum5, popsum, denssum]
            for j in range(0,5):
                df3.set_value(index, listcols[j], listvals[j] / weightpopsum)
            df3.set_value(index, listcols[5], np.log10(listvals[5] / weightsum))
            df3.set_value(index, listcols[6], np.log10(listvals[6] / weightsum))
                #print(index, i, listcols[i], listvals[i] / weightsum)
                #df3.set_value(index, 'growthweight1', growthsum1 / weightsum)
        if(weightsum != 0.0):
            print(i, len(grouplist), popsum / weightsum)
        else:
            print(grouplist[i])
    return df3

def calc_dist(lat1, lon1, lat2, lon2):
    #Calculates distance between two points on the globe.  This is essentially copied from here: https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points
    #Returns distance in kilometers
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlat = lat1 - lat2
    dlon = lon1 - lon2
    #print(lat1, lat2, dlat)
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371.0
    return c*r

def vary_sigma(df, outlist = [(0, 10)]):
    #Runs engineer clusters multiple times with different values of sigma:
    df2 = df.copy()
    df02 = engineer_clusters(df2, outlist = outlist, sigma = 20, flag = True)
    df02.to_csv('../DATA/dfclu_020b.csv')
    df05 = engineer_clusters(df02, outlist = outlist, sigma = 50, flag = False)
    df05.to_csv('../DATA/dfclu_050b.csv')
    df10 = engineer_clusters(df05, outlist = outlist, sigma = 100, flag = False)
    df10.to_csv('../DATA/dfclu_100b.csv')
    df20 = engineer_clusters(df10, outlist = outlist, sigma = 200, flag = False)
    df20.to_csv('../DATA/dfclu_200b.csv')
    return df20

if __name__ == '__main__':
    #Read in data, and create the features:
    df = pd.read_csv('../DATA/allsplits.csv')
    #If you don't want the cluster data, then just do this instead:
    #df = pd.read_csv('../DATA/year_rawfeature.csv')
    #and then for the "feature_engineer" step, change the name to df_feature
    dfy = pd.read_csv('../DATA/y1_df.csv')

    df_feat = feature_engineer(df)

    #Add in the cluster features:
    #df_feature = engineer_clusters(df_feat)
    outlist = find_outliers(df_feat)
    df_feature = vary_sigma(df_feat, outlist)

    ################
    ###### This next section is for the version without clusters.  Skip ahead to the part labeled "Cluster Section" if you're working with the version of the data with clusters.
    ################
    """
    #Now select the years to look at:
    myyears = [2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012]
    #Everything after 2012 is provisionally going to be "test data, that I'm not going to look at."
    X2000s = select_years(df_feature, myyears)
    y2000s = select_years(dfy, myyears)

    #Now select the columns:
    mylist = ['GEOID', 'YEAR', 'POP_GROWTH_P1', 'POP_GROWTH_P2', 'POP_GROWTH_P3', 'POP_GROWTH_P4', 'POP_GROWTH_P5', 'INTPTLAT', 'INTPTLONG', 'age_cohort1', 'age_cohort2', 'age_cohort3', 'age_cohort4', 'age_cohort5', 'age_cohort6', 'age_cohort7', 'age_cohort8', 'age_cohort9', 'age_cohort10', 'age_cohort11', 'age_cohort12', 'age_cohort13', 'age_cohort14', 'age_cohort15', 'age_cohort16', 'age_cohort17', 'age_cohort18', 'logpop', 'logland', 'logwater', 'fem_frac']
    X2000s_df = select_cols(X2000s, mylist)

    ### Some more EDA:
    X2000s_df.hist(bins=30, figsize=(8,10))
    plt.savefig('histograms2000s.eps')

    #Let's first combine the feature matrix with the target:
    all2000s_df = pd.concat([X2000s_df, y2000s], axis = 1)

    #Correlation matrix:
    all2000s_df.corr().to_csv('../DATA/all2000s_corr.csv')

    #OK, now output the file:
    X2000s_df.to_csv('../DATA/X2000s_df.csv')
    y2000s.to_csv('../DATA/y2000s_df.csv')

    #Here's my attempt at purging outliers:
    X2000s_nout, y2000s_nout, setlist = kill_outliers(X2000s_df, y2000s)
    X2000s_nout.to_csv('../DATA/X2000s_nout.csv')
    y2000s_nout.to_csv('../DATA/y2000s_nout.csv')"""

    ################
    ###### Cluster Section, part 1:
    ################
    #Now let's try the version with clusters:
    #Now select the years to look at:
    """myyears = [2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014]
    #2015-16 is provisionally going to be "test data, that I'm not going to look at."
    X2000s = select_years(df_feature, myyears)
    y2000s = select_years(dfy, myyears)
    #Now select the columns:
    mylist = ['GEOID', 'YEAR', 'POP_GROWTH_P1', 'POP_GROWTH_P2', 'POP_GROWTH_P3', 'POP_GROWTH_P4', 'POP_GROWTH_P5', 'INTPTLAT', 'INTPTLONG', 'age_cohort1', 'age_cohort2', 'age_cohort3', 'age_cohort4', 'age_cohort5', 'age_cohort6', 'age_cohort7', 'age_cohort8', 'age_cohort9', 'age_cohort10', 'age_cohort11', 'age_cohort12', 'age_cohort13', 'age_cohort14', 'age_cohort15', 'age_cohort16', 'age_cohort17', 'age_cohort18', 'logpop', 'logland', 'logwater', 'fem_frac', 'logpop_clu', 'logland_clu', 'pop_growth_clu1', 'pop_growth_clu2', 'growthweight1', 'growthweight2', 'growthweight3', 'growthweight4', 'growthweight5', 'popweight', 'denseweight', 'cluster', 'split']
    X2000s_df = select_cols(X2000s, mylist)

    ### Some more EDA:
    #Shrink font size:
    matplotlib.rcParams.update({'font.size': 6})
    X2000s_df.hist(bins=30, figsize=(10,12))
    plt.savefig('histograms2000club.eps')

    #Let's first combine the feature matrix with the target:
    all2000s_df = pd.concat([X2000s_df, y2000s], axis = 1)

    #Correlation matrix:
    all2000s_df.corr().to_csv('../DATA/all2000s_corr_club.csv')
    # A couple of notes from that:
    # -On the cluster level, pop growth from last year is even more strongly correlated with pop growth over last two years than the same two quantities are correlated with each other on the individual county level.
    # -The cluster past population growth isn't quite as strongly correlated with future county population growth as county past population growth is.

    #Here's my attempt at purging outliers:
    X2000s_nout, y2000s_nout, setlist = kill_outliers(X2000s_df, y2000s)
    X2000s_nout.to_csv('../DATA/X2000s_nout_club.csv')
    y2000s_nout.to_csv('../DATA/y2000s_nout_club.csv')"""


    ################
    ###### Cluster Section, part 2:
    ################
    #Redoing the cluster section, to incorporate the Gaussian smoothing of space around the counties....
    #Now select the years to look at:
    myyears = [2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014]
    #2015-16 is provisionally going to be "test data, that I'm not going to look at."
    X2000s = select_years(df_feature, myyears)
    y2000s = select_years(dfy, myyears)
    #Now select the columns:
    mylist = ['GEOID', 'YEAR', 'POP_GROWTH_P1', 'POP_GROWTH_P2', 'POP_GROWTH_P3', 'POP_GROWTH_P4', 'POP_GROWTH_P5', 'INTPTLAT', 'INTPTLONG', 'age_cohort1', 'age_cohort2', 'age_cohort3', 'age_cohort4', 'age_cohort5', 'age_cohort6', 'age_cohort7', 'age_cohort8', 'age_cohort9', 'age_cohort10', 'age_cohort11', 'age_cohort12', 'age_cohort13', 'age_cohort14', 'age_cohort15', 'age_cohort16', 'age_cohort17', 'age_cohort18', 'logpop', 'logland', 'logwater', 'fem_frac', 'logpop_clu', 'logland_clu', 'pop_growth_clu1', 'pop_growth_clu2', 'growthweight120', 'growthweight220', 'growthweight320', 'growthweight420', 'growthweight520', 'popweight20', 'denseweight20', 'growthweight150', 'growthweight250', 'growthweight350', 'growthweight450', 'growthweight550', 'popweight50', 'denseweight50', 'growthweight1100', 'growthweight2100', 'growthweight3100', 'growthweight4100', 'growthweight5100', 'popweight100', 'denseweight100', 'growthweight1200', 'growthweight2200', 'growthweight3200', 'growthweight4200', 'growthweight5200', 'popweight200', 'denseweight200', 'cluster', 'split']
    X2000s_df = select_cols(X2000s, mylist)
    #Shrink font size:
    matplotlib.rcParams.update({'font.size': 6})
    X2000s_df.hist(bins=30, figsize=(12,15))
    plt.savefig('histograms2000clug.eps')
    #Let's first combine the feature matrix with the target:
    all2000s_df = pd.concat([X2000s_df, y2000s], axis = 1)

    #Correlation matrix:
    all2000s_df.corr().to_csv('../DATA/all2000s_corr_cluc.csv')
    #OK, looks like the versions of the cluster stats that are aggregated across the entire cluster have less correlation with individual county population growth than the versions that are Gaussian-smoothed.  So I'm going to ditch the former and keep the latter.
    newlist = ['GEOID', 'YEAR', 'POP_GROWTH_P1', 'POP_GROWTH_P2', 'POP_GROWTH_P3', 'POP_GROWTH_P4', 'POP_GROWTH_P5', 'INTPTLAT', 'INTPTLONG', 'age_cohort1', 'age_cohort2', 'age_cohort3', 'age_cohort4', 'age_cohort5', 'age_cohort6', 'age_cohort7', 'age_cohort8', 'age_cohort9', 'age_cohort10', 'age_cohort11', 'age_cohort12', 'age_cohort13', 'age_cohort14', 'age_cohort15', 'age_cohort16', 'age_cohort17', 'age_cohort18', 'logpop', 'logland', 'logwater', 'fem_frac', 'growthweight120', 'growthweight220', 'growthweight320', 'growthweight420', 'growthweight520', 'popweight20', 'denseweight20', 'growthweight150', 'growthweight250', 'growthweight350', 'growthweight450', 'growthweight550', 'popweight50', 'denseweight50', 'growthweight1100', 'growthweight2100', 'growthweight3100', 'growthweight4100', 'growthweight5100', 'popweight100', 'denseweight100', 'growthweight1200', 'growthweight2200', 'growthweight3200', 'growthweight4200', 'growthweight5200', 'popweight200', 'denseweight200', 'cluster', 'split']
    X2000s2_df = select_cols(X2000s, newlist)
    outcols = ['POP_GROWTH_P1', 'POP_GROWTH_P2', 'POP_GROWTH_P3', 'POP_GROWTH_P4', 'POP_GROWTH_P5']
    X2000s_nout, y2000s_nout, setlist = kill_more_outliers(X2000s2_df, y2000s, outcols)
    #After some trial and error, just going to go with killing outliers on those five columns.  I've edited engineer_clusters so that it can better deal with outliers in the smoothing, so it should be OK.
    X2000s_nout.to_csv('../DATA/X2000s_nout_clug.csv')
    y2000s_nout.to_csv('../DATA/y2000s_nout_clug.csv')
