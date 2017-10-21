# CountyPopulationGrowth

The goal is to construct a model that predicts the population growth of each individual county in the United States based on current / past demographic information at the county level, starting at a time horizon of just 1 year into the future, but with the goal of expanding it out to greater time intervals.

While the immediate variable of interest is population growth, the architecture of the model could be re-used for other purposes.  E.g., businesses might be interested in marketing nationwide, and might be interested in predicting some other variable from a model that takes as inputs demographic information from every individual county in the country.

### Data

The physical county parameters downloaded under the 'counties' link here:

https://www.census.gov/geo/maps-data/data/gazetteer2016.html

This was downloadable as a text file, and I've now converted it into a CSV file.
The annual demographic parameters downloaded from here:

https://www2.census.gov/programs-surveys/popest/datasets/

These are in two separate files.  One for the decade beginning in 2000, and another for the decade beginning in 2010.

### Code

In the SRC directory, there are four pieces of code (executed in this order):

make_nhgis_csv.py is just a few lines long, and it was used to convert the geographic data from text format to csv format.

clean_yearly.py reformats the data into something usable, and outputs new csv files with the cleaned data.  It was also used for some EDA, which I've now commented out.

yearly_samples.py does some feature engineering and outputs a revised set of csv files with selected features and for selected years.  I did a bit more EDA at this stage.

runmodel.py does the fitting.  Lasso, Ridge, and Random Forest all fit on data from selected years.  Earlier trials are commented out, and latest run fitting on a 10 year window of 2002-2011 is shown at the bottom where I write "Here's the code for MVP".  For this exercise, the 2012 data is treated as "test" data.  (The real test data is the set for 2013 and later, which I'm saving to test on at the end of the project.)

### MVP results

The output of runmodel.py gives summary statistics at the end for the "test" predictions for 2012-2013 growth by county ( FIGS/mvpyear2012.txt ).  As has happened in every run so far, Random Forest gives a better R^2 value in the cross-validation than the linear regression models.  In this run, the R^2 is 0.465 (compared to 0.428 for both Lasso and Ridge).  The two most important features are population growth from last year and population growth over the past two years, and those dominate.  Next is current population, and then the number of people from the oldest age cohorts (the latter being negatively correlated with population growth, as shown by the Lasso coefficients.)

FIGS/2012predvsactual.eps shows the actual 2012-2013 population growth (y-axis) vs. population growth predicted by the model (x-axis).

**Note that population growth here is expressed as log(population next year / population this year), rather than percentage change.**
