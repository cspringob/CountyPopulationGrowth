import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

def clean2000(df):
    #This is for converting the year 2000-2009 data into a usable dataframe:
    df['GEOID'] = 1000 * df['STATE'] + df['COUNTY']
    df = df.drop('SUMLEV', 1)
    df = df.drop('COUNTY', 1)
    df = df.drop('ESTIMATESBASE2000', 1)
    df = df.drop('CENSUS2010POP', 1)
    df2 = pd.melt(df, id_vars=['STATE', 'STNAME', 'CTYNAME', 'SEX', 'AGEGRP', "GEOID"],var_name="Date", value_name="POPULATION")
    df2['YEAR'] = df2['Date'].str[-4:]
    df2 = df2.drop('Date', 1)
    df2.convert_objects(convert_numeric=True)
    df2['SEX-AGE'] = 100 * df2['SEX'] + df2['AGEGRP']
    #Note: for sex...0 = tot, 1 = male, 2 = female
    df2 = df2.drop('SEX', 1)
    df2 = df2.drop('AGEGRP', 1)
    df3 = df2.pivot_table(values='POPULATION', index=['GEOID', 'YEAR'], columns='SEX-AGE')
    df3.reset_index(level=df3.index.names, inplace=True)
    #fill in the nans, and convert floats to ints:
    df3 = df3.fillna(0)
    columns = list(df3)
    df4 = df3.astype(int)
    return df4
    #Purge the extra counties:
    #df5 = df4[df4['GEOID'] not in purge_list]
    #return df5

def clean2010(df):
    #This is for converting the year 2010- data into a usable dataframe:
    df['GEOID'] = 1000 * df['STATE'] + df['COUNTY']
    df2 = df[['GEOID', 'YEAR', 'AGEGRP', 'TOT_POP', 'TOT_MALE', 'TOT_FEMALE']].copy()
    df2.AGEGRP = df2.AGEGRP.astype(str)
    df3 = df2.pivot_table(values=['TOT_POP', 'TOT_MALE', 'TOT_FEMALE'], index=['GEOID', 'YEAR'], columns='AGEGRP')
    #Flatten the column indexing:
    df3.reset_index(level=df3.index.names, inplace=True)
    df3.columns = [''.join(str(col)).strip() for col in df3.columns.values]
    df4 = df3.rename(index=str, columns={"('GEOID', '')": 'GEOID', "('YEAR', '')": 'YEAR'})
    for i in range(0,19):
        namepop = ''.join(["('TOT_POP', '", str(i), "')"])
        namemal = ''.join(["('TOT_MALE', '", str(i), "')"])
        namefem = ''.join(["('TOT_FEMALE', '", str(i), "')"])
        df4 = df4.rename(index=str, columns = {namepop: i, namemal: 100 + i, namefem: 200 + i})
    df5 = df4[df4['YEAR'] > 3]
    df5['YEAR'] = df5['YEAR'] + 2007
    return df5
    #df6 = df5[df5['GEOID'] not in purge_list]
    #return df6

def compare_counties(df1, df2):
    #Compare the two tables to identify which counties show up in one but not the other:
    #df1['GEOID'] = 1000 * df1['STATE'] + df1['COUNTY']
    #df2['GEOID'] = 1000 * df2['STATE'] + df2['COUNTY']
    df1['tuple'] = list(zip(df1['CTYNAME'], df1['GEOID']))
    df2['tuple'] = list(zip(df2['CTYNAME'], df2['GEOID']))
    print(set(df1['tuple'].unique()) - set(df2['tuple'].unique()))
    print(set(df2['tuple'].unique()) - set(df1['tuple'].unique()))
    df1 = df1.drop('tuple', 1)
    df2 = df2.drop('tuple', 1)
    return None

def calc_growth(df):
    #Calculates past and future population growth.  Adds past growth as a column in the dataframe, and also returns future growth as the target dataframe.
    count = 0
    myarr = np.zeros(len(df))
    for i in range(0,len(df)/17):
        for j in range(0,17):
            if(j == 0):
                myarr[count] = 0
            else:
                myarr[count] = math.log10(float(df[0][count]) / float(df[0][count-1]))
            #print(count,myarr[count], df3[0][count])
            count = count + 1
    #Now make a list of future population growth:
    myarr_f = np.zeros(len(df))
    for i in range(0,len(df)-1):
        myarr_f[i] = myarr[i+1]
    df['POP_GROWTH_P'] = myarr
    #Check to make sure we didn't make any mistakes:
    df_checklist = df[(df['POP_GROWTH_P'] == 0.0) & (df['YEAR'] > 2000)]
    #Make a dataframe with the target values:
    df2 = df[['GEOID', 'YEAR']].copy()
    df2['POP_GROWTH_F'] = myarr_f
    return df, df2, df_checklist

def calc_growth_lt(df):
    #Calculates past and future population growth over 1-5 years.  Adds past growth numbers as columns in the dataframe, and also returns 1 and 5 year future growth as the target dataframe.
    myarr = np.zeros((5, len(df)))
    for k in range(0,5):
        count = 0
        for i in range(0,len(df)/17):
            for j in range(0,17):
                if(j < k + 1):
                    myarr[k][count] = 0
                else:
                    myarr[k][count] = math.log10(float(df[0][count]) / float(df[0][count-(k + 1)]))
                #print(count,myarr[count], df3[0][count])
                count = count + 1
    #Now make a list of future population growth:
    myarr_f = np.zeros((5, len(df)))
    for k in range(0,5):
        for i in range(0,len(df)-(k+1)):
            myarr_f[k][i] = myarr[k][i + k + 1]
        df[''.join(['POP_GROWTH_P', str(k+1)])] = myarr[k]
    #Make dataframes with the target values for 1 year and 5 years:
    df2 = df[['GEOID', 'YEAR']].copy()
    df2['POP_GROWTH_F1'] = myarr_f[0]
    df3 = df[['GEOID', 'YEAR']].copy()
    df3['POP_GROWTH_F5'] = myarr_f[4]
    return df, df2, df3

def select_years(df, yearlist):
    #Return a dataframe that only includes rows from the relevant years in "yearlist":
    dfout = df[df_feature['YEAR'].isin(yearlist)].copy()
    return dfout

def select_cols(df, mylist):
    #Returns a dataframe with only the selected columns
    dfout = df[mylist].copy()
    return dfout

def makelog(df):
    #Creates logarithmic versions of selected columns:
    df['logpop'] = np.log10(df[0].astype(float))
    df['logland'] = np.log10(df['ALAND_SQMI'])
    #Need to set low values of water square mileage to a positive value first:
    df.loc[df['AWATER_SQMI'] < 0.01, 'AWATER_SQMI'] = 0.01
    df['logwater'] = np.log10(df['AWATER_SQMI'])
    df['logpopdens'] = np.log10(df['pop_density'])
    return df

def earth_is_round(df):
    #This converts a positive longitude into negative longitude:
    df.loc[df['INTPTLONG'] > 0, 'INTPTLONG'] = df['INTPTLONG'] - 360.0
    return df

if __name__ == '__main__':
    #Read in the datafames:
    df0 = pd.read_csv('../DATA/co-est00int-agesex-5yr.csv')
    df1 = pd.read_csv('../DATA/cc-est2016-alldata.csv')
    dfgeo = pd.read_csv('../DATA/2016_Gaz_counties_national.csv')
    #Fix up the geographic one:
    dfgeo = dfgeo.drop('Unnamed: 0', 1)
    dfgeo = dfgeo.rename(index=str, columns={'INTPTLONG                                                                                                               ': 'INTPTLONG'})

    #Convert them into a sensible format:
    df0_clean = clean2000(df0)
    df1_clean = clean2010(df1)

    #Compare the two tables to see which counties are in one but not the other:
    compare_counties(df0, df1)
    """Here's the output:
    set([('Wade Hampton Census Area', 2270), ('Bedford city', 51515), ('Petersburg Census Area', 2195), ('La Salle Parish', 22059), ('Shannon County', 46113)])
    set([('Kusilvak Census Area', 2158), ('Oglala Lakota County', 46102), ('LaSalle Parish', 22059), ('Petersburg Borough', 2195)])

    "Petersburg Borough / Petersburg Census Area" look like the same thing under a different name.  Likewise, "La Salle Parish / LaSalle Parish" is the same thing, just a difference in spaces.

    In df0, Bedford City is unique, as are Shannon County and Wade Hampton Census Area.  In df1, Kusilvak Census Area and Oglala Lakota County are also unique.  So purge these GEOIDs:
    2270, 46113, 51515
    2158, 46102
    """
    #Purge these counties:
    df0_cleaner = df0_clean[(df0_clean['GEOID'] != 2270) & (df0_clean['GEOID'] != 46113) & (df0_clean['GEOID'] != 51515)]
    df1_cleaner = df1_clean[(df1_clean['GEOID'] != 2158) & (df1_clean['GEOID'] != 46102)]

    #Merge them into one table:
    df_comb = pd.concat([df0_cleaner, df1_cleaner]).sort_values(['GEOID', 'YEAR'], ascending = [True, True])
    df_comb = df_comb.reset_index(drop=True)

    #df_rawfeature, df_y, df_checklist = calc_growth(df_comb)
    df_rawfeature, df_y1, df_y5 = calc_growth_lt(df_comb)

    #There are 137 rows in the df_checklist file.  I investigated a handful of them, by just comparing against the same county in the previous year.  It looks like those are just cases where there happened to be no change from one year to the next, rather than a problem with the code.

    #Now, merge with the geographic table:
    df_feature = pd.merge(df_rawfeature,dfgeo, how='inner', left_on='GEOID', right_on='GEOID')

    df_feature = earth_is_round(df_feature)

    df_feature.to_csv('../DATA/year_rawfeature.csv')
    df_y1.to_csv('../DATA/y1_df.csv')
    df_y5.to_csv('../DATA/y5_df.csv')

    #This is the initial round of feature addition.  Everything before this was used to get the initial dataframes:

    #Now, add a few features to the dataset:
    #Fraction of the population under 5 years old:
    df_feature['under5_frac'] = df_feature[1].astype(float) / df_feature[0].astype(float)
    #Fraction of the population under 20 years old:
    df_feature['under20_frac'] = (df_feature[1].astype(float) + df_feature[2].astype(float) + df_feature[3].astype(float) + df_feature[4].astype(float)) / df_feature[0].astype(float)
    #Fraction of the population over 65 years old:
    df_feature['over65_frac'] = (df_feature[14].astype(float) + df_feature[15].astype(float) + df_feature[16].astype(float) + df_feature[17].astype(float) + df_feature[18].astype(float)) / df_feature[0].astype(float)
    #Fraction of the population that's female:
    df_feature['fem_frac'] = df_feature[200].astype(float) / df_feature[0].astype(float)
    #Population density:
    df_feature['pop_density'] = df_feature[0].astype(float) / df_feature['ALAND_SQMI']

    #########
    # OK, now some EDA:
    #########
    #Let's work with just the year 2001 as our initial training set:
    X2001 = select_years(df_feature, [2001])
    y2001 = select_years(df_y, [2001])

    X2001.describe().T
    mylist = ['GEOID', 'YEAR', 0, 'POP_GROWTH_P', 'ALAND_SQMI', 'AWATER_SQMI', 'INTPTLAT', 'INTPTLONG', 'under5_frac', 'under20_frac', 'over65_frac', 'fem_frac', 'pop_density']
    X2001_df = select_cols(X2001, mylist)

    #Let's just look at the geographic distribution of counties:
    plt.scatter(X2001_df['INTPTLONG'], X2001_df['INTPTLAT'])
    plt.savefig('latlonplot.eps')

    #Looks like we might have some discrepant data across the Atlantic Ocean.  Let's select everything that's outside the continental US:
    X2001[(X2001['INTPTLAT'] > 50) | (X2001['INTPTLAT'] < 25) | (X2001['INTPTLONG'] > -50) | (X2001['INTPTLONG'] < -130)][['GEOID', 'NAME', 'USPS', 'INTPTLAT', 'INTPTLONG']]

    """Here's the output:
          GEOID                               NAME USPS   INTPTLAT   INTPTLONG
    1140   2013             Aleutians East Borough   AK  55.245044 -161.997477
    1157   2016         Aleutians West Census Area   AK  51.959447  178.338813
    1174   2020             Anchorage Municipality   AK  61.174250 -149.284329
    1191   2050                 Bethel Census Area   AK  60.929141 -160.152625
    1208   2060                Bristol Bay Borough   AK  58.730149 -156.996679
    1225   2068                     Denali Borough   AK  63.681106 -150.026544
    1242   2070             Dillingham Census Area   AK  60.299091 -158.097320
    1259   2090       Fairbanks North Star Borough   AK  64.692317 -146.601733
    1276   2100                     Haines Borough   AK  59.098771 -135.576936
    1293   2105          Hoonah-Angoon Census Area   AK  58.403336 -135.884909
    1310   2110            Juneau City and Borough   AK  58.372910 -134.178445
    1327   2122            Kenai Peninsula Borough   AK  60.366668 -152.322283
    1344   2130          Ketchikan Gateway Borough   AK  55.449938 -131.106685
    1361   2150              Kodiak Island Borough   AK  57.270775 -153.953863
    1378   2164         Lake and Peninsula Borough   AK  58.108693 -156.414141
    1395   2170          Matanuska-Susitna Borough   AK  62.279075 -149.617562
    1412   2180                   Nome Census Area   AK  64.783686 -164.188912
    1429   2185                North Slope Borough   AK  69.449343 -153.472830
    1446   2188           Northwest Arctic Borough   AK  67.005066 -160.021086
    1463   2195                 Petersburg Borough   AK  57.112458 -133.008598
    1480   2198  Prince of Wales-Hyder Census Area   AK  55.682773 -133.162389
    1497   2220             Sitka City and Borough   AK  57.193204 -135.367396
    1514   2230               Skagway Municipality   AK  59.575097 -135.335418
    1531   2240    Southeast Fairbanks Census Area   AK  63.864997 -143.218628
    1548   2261         Valdez-Cordova Census Area   AK  61.349840 -145.023141
    1565   2275          Wrangell City and Borough   AK  56.180779 -132.026788
    1582   2282           Yakutat City and Borough   AK  60.019872 -140.414566
    1599   2290          Yukon-Koyukuk Census Area   AK  65.375716 -151.577878
    9266  15001                      Hawaii County   HI  19.597764 -155.502443
    9283  15003                    Honolulu County   HI  21.461365 -158.201974
    9300  15005                     Kalawao County   HI  21.218764 -156.974010
    9317  15007                       Kauai County   HI  22.012038 -159.705965
    9334  15009                        Maui County   HI  20.855931 -156.601550

    Looks like Aleutians West Census Area is in the Eastern Hemisphere, so I'll have to watch out for that."""

    #I guess I can fix that right now:
    X2001_df = earth_is_round(X2001_df)

    #Now let's plot histograms of everything:
    X2001_df.hist(bins=30, figsize=(8,10))
    plt.savefig('histograms.eps')
    #Based on this, I'm thinking that these parameters should be expressed in log units:
    #population, land area, water area, population density
    #X2001_logs_df = makelog(X2001_df)
    #Problem.....some of the counties have zero values for AWATER_SQMI, so logarithm doesn't work.  Here's a list of those with low values:
    """
           GEOID  YEAR      0  POP_GROWTH_P  ALAND_SQMI  AWATER_SQMI   INTPTLAT  \
    15675  20071  2001   1524     -0.001706     778.404        0.000  38.480404
    47108  48501  2001   7299      0.001490     799.715        0.013  33.162398
    49692  51610  2001  10550      0.004510       2.051        0.000  38.884722
    49862  51685  2001  10777      0.019450       2.539        0.000  38.768945
    """
    #So only three counties with zero.  Let's just set everything below 0.01 to 0.01.  I've changed makelog to do this now.  Here's the redo:

    X2001_logs_df = makelog(X2001_df)

    #Now redo histograms:
    X2001_logs_df.hist(bins=30, figsize=(8,10))
    plt.savefig('histograms_log.eps')

    #How about the correlation matrix and scatter matrix?
    #Let's first combine the feature matrix with the target:
    all2001_df = pd.concat([X2001_logs_df, y2001], axis = 1)

    #Correlation matrix:
    all2001_df.corr().to_csv('../DATA/all2001_corr.csv')

    #Scatter matrix:
    myvars = ['POP_GROWTH_P', 'INTPTLAT', 'INTPTLONG', 'under5_frac', 'under20_frac', 'over65_frac', 'fem_frac', 'logpop', 'logland', 'logwater', 'logpopdens', 'POP_GROWTH_F']
    scatter_matrix(all2001_df[myvars], figsize=(20,20), diagonal = 'kde')
    plt.savefig('scattermatrix.eps')

    """A few conclusions from this:
    -The logarithmic versions of each of the variables that I logarithmized seem better than the non-log versions.  Correlations w/ the target are better, etc.
    -We have the triplet log(pop), log(area), log(pop density), which are a triplet.  You know two, then you know the third.  So I'll ditch log(pop density), since the other two have the lowest pairwise correlation between each other.
    -The strongest correlations with pop growth_f are:
    pop_growth_p
    over65_frac (negative correlation)
    logpop
    under5_frac
    latitude (negative correlation)

    Weakest correlations are:
    logland
    fem_frac
    longitude
    under20_frac
    logwater

    Other correlations that are greater than 0.5:
    over65_frac is negatively correlated with under5_frac: -0.61
    under5_frac is positively correlated with under20_frac: 0.81
    That correlation is so strong that I think I'll drop under20_frac.
    However, later on, I might add in fractions for every age cohort.
    """
    newlist = ['GEOID', 'YEAR', 'POP_GROWTH_P', 'INTPTLAT', 'INTPTLONG', 'under5_frac', 'over65_frac', 'fem_frac', 'logpop', 'logland', 'logwater']
    finalX_2001 = select_cols(X2001_logs_df, newlist)

    #finalX_2001.to_csv('../DATA/X2001_df.csv')
    #y2001.to_csv('../DATA/y2001_df.csv')
