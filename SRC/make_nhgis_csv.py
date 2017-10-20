import pandas as pd

#This program converts the text file version of the geographic data into a csv file.

df = pd.read_csv('../DATA/2016_Gaz_counties_national.txt', sep='\t')
df2 = df.round(6)
df2.to_csv('../DATA/2016_Gaz_counties_national.csv')
