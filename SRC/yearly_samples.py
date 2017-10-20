import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

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

if __name__ == '__main__':
    #Read in data, and create the features:
    df = pd.read_csv('../DATA/year_rawfeature.csv')
    dfy = pd.read_csv('../DATA/y1_df.csv')
    df_feature = feature_engineer(df)

    #Now select the years to look at:
    myyears = [2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012]
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

    #Scatter matrix:
    """myvars = ['POP_GROWTH_P1', 'POP_GROWTH_P2', 'POP_GROWTH_P3', 'POP_GROWTH_P4', 'POP_GROWTH_P5', 'INTPTLAT', 'INTPTLONG', 'age_cohort1', 'age_cohort2', 'age_cohort3', 'age_cohort4', 'age_cohort5', 'age_cohort6', 'age_cohort7', 'age_cohort8', 'age_cohort9', 'age_cohort10', 'age_cohort11', 'age_cohort12', 'age_cohort13', 'age_cohort14', 'age_cohort15', 'age_cohort16', 'age_cohort17', 'age_cohort18', 'logpop', 'logland', 'logwater', 'fem_frac', 'POP_GROWTH_F1']
    scatter_matrix(all2000s_df[myvars], figsize=(20,20), diagonal = 'kde')
    plt.savefig('scattermatrix2000s.eps')"""
    #Never mind.  Making a scatter matrix with this many parameters takes way too long.

    #OK, now output the file:
    X2000s_df.to_csv('../DATA/X2000s_df.csv')
    y2000s.to_csv('../DATA/y2000s_df.csv')

    #Here's my attempt at purging outliers:
    X2000s_nout, y2000s_nout, setlist = kill_outliers(X2000s_df, y2000s)
    X2000s_nout.to_csv('../DATA/X2000s_nout.csv')
    y2000s_nout.to_csv('../DATA/y2000s_nout.csv')
