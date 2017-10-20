import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import math

def run_regression(model, X, y, alphas):
    grid = GridSearchCV(estimator = model, param_grid = dict(alpha=alphas), cv = 10)
    grid.fit(X, y)
    return grid

def run_regressionlist(model, X, y, alphas):
    scoring = {'r2': 'r2', 'neg_mean_squared_error': 'neg_mean_squared_error'}
    grid = GridSearchCV(estimator = model, param_grid = dict(alpha=alphas), cv = 10, scoring = scoring, refit = 'r2')
    grid.fit(X, y)
    return grid

def run_forest(model, X, y, parameters):
    grid = GridSearchCV(model, parameters, cv = 5, verbose = 100)
    grid.fit(X,y)
    return grid

def run_forestlist(model, X, y, parameters):
    scoring = {'r2': 'r2', 'neg_mean_squared_error': 'neg_mean_squared_error'}
    grid = GridSearchCV(model, parameters, cv = 5, scoring = scoring, refit = 'r2', verbose = 100)
    grid.fit(X,y)
    return grid

def make_array(grid):
    #Plots r squared against alpha.  The input "grid" is actually grid.grid_scores_
    alphaarr = np.zeros(len(grid))
    rsqarr = np.zeros(len(grid))
    for i in range(0,len(grid)):
        alphaarr[i] = math.log10(grid[i][0]['alpha'])
        rsqarr[i] = grid[i][1]
    return alphaarr, rsqarr

def select_years(df, yearlist):
    #Return a dataframe that only includes rows from the relevant years in "yearlist":
    dfout = df[df['YEAR'].isin(yearlist)].copy()
    return dfout

def loop_gridsearch(Xdf, ydf, yearlists, featurelist, alphas, forest_par, filename):
    #Makes blocks of data from yearlists, and runs through lasso, ridge, and random forest on all of them.
    f1 = open(filename,"w")
    lasso = Lasso()
    ridge = Ridge()
    forest = RandomForestRegressor()
    las_list = []
    rid_list = []
    for_list = []
    for i in range(0,len(yearlists)):
        nextyear = [yearlists[i][-1] + 1]
        #print(nextyear)
        Xt = select_years(Xdf, yearlists[i])
        yt = select_years(ydf, yearlists[i])
        Xt2 = select_years(Xdf, nextyear)
        yt2 = select_years(ydf, nextyear)
        X = Xt[featurelist]
        y = yt['POP_GROWTH_F1']
        Xtest = Xt2[featurelist]
        ytest = yt2['POP_GROWTH_F1']
        scaler = StandardScaler()
        Xscale = scaler.fit_transform(X)
        Xscaletest = scaler.transform(Xtest)
        las_list.append(run_regressionlist(lasso, Xscale, y, alphas))
        rid_list.append(run_regressionlist(ridge, Xscale, y, alphas))
        for_list.append(run_forestlist(forest, Xscale, y, forest_par))
        mymean = np.mean(ytest.values)
        myrms = np.mean(ytest.values * ytest.values)
        laspredict = las_list[i].predict(Xscaletest)
        lasoffset = np.mean(ytest.values - laspredict)
        lasrms = np.mean((ytest.values - laspredict) * (ytest.values - laspredict))
        ridpredict = rid_list[i].predict(Xscaletest)
        ridoffset = np.mean(ytest.values - ridpredict)
        ridrms = np.mean((ytest.values - ridpredict) * (ytest.values - ridpredict))
        forpredict = for_list[i].predict(Xscaletest)
        foroffset = np.mean(ytest.values - forpredict)
        forrms = np.mean((ytest.values - forpredict) * (ytest.values - forpredict))
        print(nextyear[0], len(yearlists[i]), las_list[i].best_score_, las_list[i].cv_results_['mean_test_neg_mean_squared_error'][las_list[i].best_index_], rid_list[i].best_score_, rid_list[i].cv_results_['mean_test_neg_mean_squared_error'][rid_list[i].best_index_], for_list[i].best_score_, for_list[i].cv_results_['mean_test_neg_mean_squared_error'][for_list[i].best_index_], lasoffset, lasrms, ridoffset, ridrms, foroffset, forrms, mymean, myrms, file = f1)
        print(nextyear[0], len(yearlists[i]), las_list[i].best_score_, las_list[i].cv_results_['mean_test_neg_mean_squared_error'][las_list[i].best_index_], rid_list[i].best_score_, rid_list[i].cv_results_['mean_test_neg_mean_squared_error'][rid_list[i].best_index_], for_list[i].best_score_, for_list[i].cv_results_['mean_test_neg_mean_squared_error'][for_list[i].best_index_], lasoffset, lasrms, ridoffset, ridrms, foroffset, forrms, mymean, myrms)
    las_list.append(yearlists)
    rid_list.append(yearlists)
    for_list.append(yearlists)
    f1.close()
    return las_list, rid_list, for_list

if __name__ == '__main__':
    #Here's my trial with just the 2001 data:
    """Xdf = pd.read_csv('../DATA/X2001_df.csv')
    ydf = pd.read_csv('../DATA/y2001_df.csv')
    X = Xdf[['POP_GROWTH_P', 'INTPTLAT', 'INTPTLONG', 'under5_frac', 'over65_frac', 'fem_frac', 'logpop', 'logland', 'logwater']]
    y = ydf['POP_GROWTH_F']
    scaler = StandardScaler()
    Xscale = scaler.fit_transform(X)

    alphas_l = list(np.logspace(-6, -4, num = 100))
    lasso = Lasso()
    las_grid = run_regression(lasso, Xscale, y, alphas_l)
    alphas_r = list(np.logspace(0, 2, num = 100))
    ridge = Ridge()
    rid_grid = run_regression(ridge, Xscale, y, alphas_r)
    forest_params = {'n_estimators':[10,100,1000], 'max_features':[0.5, 0.7, 0.8, 0.9], 'min_samples_leaf':[1,3,5]}
    forest = RandomForestRegressor()
    for_grid = run_forest(forest, Xscale, y, forest_params)
    #The forest regressor took ~15 minutes to run 180 forests (5 folds * 36 hyperparameters to check).
    print(las_grid.best_score_, las_grid.best_params_, las_grid.best_estimator_.coef_)
    print(rid_grid.best_score_, rid_grid.best_params_, rid_grid.best_estimator_.coef_)
    print(for_grid.best_score_, for_grid.best_params_, for_grid.best_estimator_.feature_importances_)
    #From this, it looks like Lasso and Ridge have very different best fit values of alpha.  However, their R^2 is virtually identical.  0.45523 for Lasso and 0.45518 for Ridge.
    #But for random forest, I get an R^2 of 0.519, which is better than the other two.
    #1000 trees might be overkill though.  With 100 trees, it's 0.515, so not that much of an improvement.
    #In both of the linear regression estimators, past pop growth is most important, followed by fraction of pop over 65.  For random forest, it's past population growth followed by log(pop).

    #OK, let's try plotting R^2 against alpha:
    alph, rsq = make_array(las_grid.grid_scores_)
    plt.scatter(alph, rsq)
    plt.xlim(-6, -4)
    plt.ylim(0.4550, 0.4554)
    plt.savefig('lasso_alpha.eps')
    alphr, rsqr = make_array(rid_grid.grid_scores_)
    plt.scatter(alphr, rsqr)
    plt.xlim(0, 2)
    plt.ylim(0.4551, 0.4552)
    plt.savefig('ridge_alpha.eps')"""

    #Now I'll try running fits on the 2000-2009 data:
    Xdf = pd.read_csv('../DATA/X2000s_nout.csv')
    ydf = pd.read_csv('../DATA/y2000s_nout.csv')
    #Initially, we'll only look at 2005-2009:
    """yearlist = [2005, 2006, 2007, 2008, 2009]
    Xt = select_years(Xdf, yearlist)
    yt = select_years(ydf, yearlist)
    X = Xt[['POP_GROWTH_P1', 'POP_GROWTH_P2', 'POP_GROWTH_P3', 'POP_GROWTH_P4', 'POP_GROWTH_P5', 'INTPTLAT', 'INTPTLONG', 'age_cohort1', 'age_cohort2', 'age_cohort3', 'age_cohort4', 'age_cohort5', 'age_cohort6', 'age_cohort7', 'age_cohort8', 'age_cohort9', 'age_cohort10', 'age_cohort11', 'age_cohort12', 'age_cohort13', 'age_cohort14', 'age_cohort15', 'age_cohort16', 'age_cohort17', 'age_cohort18', 'logpop', 'logland', 'logwater', 'fem_frac']]
    y = yt['POP_GROWTH_F1']

    #All right....let's get fitting:
    scaler = StandardScaler()
    Xscale = scaler.fit_transform(X)
    alphas_l = list(np.logspace(-6, -4, num = 20))
    lasso = Lasso()
    las_grid = run_regression(lasso, Xscale, y, alphas_l)
    alphas_r = list(np.logspace(2, 6, num = 20))
    ridge = Ridge()
    rid_grid = run_regression(ridge, Xscale, y, alphas_r)
    print(las_grid.best_score_, las_grid.best_params_, las_grid.best_estimator_.coef_)
    print(rid_grid.best_score_, rid_grid.best_params_, rid_grid.best_estimator_.coef_)
    forest_params = {'n_estimators':[10,100], 'max_features':[0.5, 0.7, 0.8, 0.9], 'min_samples_leaf':[1,3,5]}
    forest = RandomForestRegressor()
    for_grid = run_forest(forest, Xscale, y, forest_params)
    print(for_grid.best_score_, for_grid.best_params_, for_grid.best_estimator_.feature_importances_)

    #OK, now let's try dropping the "pop over last N years" columns 1 by 1:
    forest_params = {'n_estimators':[100], 'max_features':[0.5, 0.7, 0.9], 'min_samples_leaf':[3,5]}
    scaler = StandardScaler()
    X_no5 = Xt[['POP_GROWTH_P1', 'POP_GROWTH_P2', 'POP_GROWTH_P3', 'POP_GROWTH_P4', 'INTPTLAT', 'INTPTLONG', 'age_cohort1', 'age_cohort2', 'age_cohort3', 'age_cohort4', 'age_cohort5', 'age_cohort6', 'age_cohort7', 'age_cohort8', 'age_cohort9', 'age_cohort10', 'age_cohort11', 'age_cohort12', 'age_cohort13', 'age_cohort14', 'age_cohort15', 'age_cohort16', 'age_cohort17', 'age_cohort18', 'logpop', 'logland', 'logwater', 'fem_frac']]
    Xscale_no5 = scaler.fit_transform(X_no5)
    forest = RandomForestRegressor()
    for_grid_no5 = run_forest(forest, Xscale_no5, y, forest_params)
    print(for_grid_no5.best_score_, for_grid_no5.best_params_, for_grid_no5.best_estimator_.feature_importances_)

    forest_params = {'n_estimators':[100], 'max_features':[0.5, 0.7], 'min_samples_leaf':[3,5]}
    scaler = StandardScaler()
    X_no4 = Xt[['POP_GROWTH_P1', 'POP_GROWTH_P2', 'POP_GROWTH_P3', 'INTPTLAT', 'INTPTLONG', 'age_cohort1', 'age_cohort2', 'age_cohort3', 'age_cohort4', 'age_cohort5', 'age_cohort6', 'age_cohort7', 'age_cohort8', 'age_cohort9', 'age_cohort10', 'age_cohort11', 'age_cohort12', 'age_cohort13', 'age_cohort14', 'age_cohort15', 'age_cohort16', 'age_cohort17', 'age_cohort18', 'logpop', 'logland', 'logwater', 'fem_frac']]
    Xscale_no4 = scaler.fit_transform(X_no4)
    forest = RandomForestRegressor()
    for_grid_no4 = run_forest(forest, Xscale_no4, y, forest_params)
    print(for_grid_no4.best_score_, for_grid_no4.best_params_, for_grid_no4.best_estimator_.feature_importances_)

    forest_params = {'n_estimators':[100], 'max_features':[0.5, 0.7], 'min_samples_leaf':[3,5]}
    scaler = StandardScaler()
    X_no3 = Xt[['POP_GROWTH_P1', 'POP_GROWTH_P2', 'INTPTLAT', 'INTPTLONG', 'age_cohort1', 'age_cohort2', 'age_cohort3', 'age_cohort4', 'age_cohort5', 'age_cohort6', 'age_cohort7', 'age_cohort8', 'age_cohort9', 'age_cohort10', 'age_cohort11', 'age_cohort12', 'age_cohort13', 'age_cohort14', 'age_cohort15', 'age_cohort16', 'age_cohort17', 'age_cohort18', 'logpop', 'logland', 'logwater', 'fem_frac']]
    Xscale_no3 = scaler.fit_transform(X_no3)
    forest = RandomForestRegressor()
    for_grid_no3 = run_forest(forest, Xscale_no3, y, forest_params)
    print(for_grid_no3.best_score_, for_grid_no3.best_params_, for_grid_no3.best_estimator_.feature_importances_)

    forest_params = {'n_estimators':[100], 'max_features':[0.5, 0.7], 'min_samples_leaf':[3,5]}
    scaler = StandardScaler()
    X_no2 = Xt[['POP_GROWTH_P1', 'INTPTLAT', 'INTPTLONG', 'age_cohort1', 'age_cohort2', 'age_cohort3', 'age_cohort4', 'age_cohort5', 'age_cohort6', 'age_cohort7', 'age_cohort8', 'age_cohort9', 'age_cohort10', 'age_cohort11', 'age_cohort12', 'age_cohort13', 'age_cohort14', 'age_cohort15', 'age_cohort16', 'age_cohort17', 'age_cohort18', 'logpop', 'logland', 'logwater', 'fem_frac']]
    Xscale_no2 = scaler.fit_transform(X_no2)
    forest = RandomForestRegressor()
    for_grid_no2 = run_forest(forest, Xscale_no2, y, forest_params)
    print(for_grid_no2.best_score_, for_grid_no2.best_params_, for_grid_no2.best_estimator_.feature_importances_)"""

    #OK, now I'll work on doing a big loop through all the data:

    featurelist = ['POP_GROWTH_P1', 'POP_GROWTH_P2', 'INTPTLAT', 'INTPTLONG', 'age_cohort1', 'age_cohort2', 'age_cohort3', 'age_cohort4', 'age_cohort5', 'age_cohort6', 'age_cohort7', 'age_cohort8', 'age_cohort9', 'age_cohort10', 'age_cohort11', 'age_cohort12', 'age_cohort13', 'age_cohort14', 'age_cohort15', 'age_cohort16', 'age_cohort17', 'age_cohort18', 'logpop', 'logland', 'logwater', 'fem_frac']
    alphas = list(np.logspace(-6, 6, num = 20))
    forest_params = {'n_estimators':[50], 'max_features':[0.3, 0.5, 0.7], 'min_samples_leaf':[5], 'n_jobs':[-1]}
    #Here are the one year lists:
    oneyearlists = [[2006], [2007], [2008], [2009], [2010], [2011]]
    las_list1, rid_list1, for_list1 = loop_gridsearch(Xdf, ydf, oneyearlists, featurelist, alphas, forest_params, 'oneyear.txt')
    #Two year lists:
    twoyearlists = [[2005, 2006], [2006, 2007], [2007, 2008], [2008, 2009], [2009, 2010], [2010, 2011]]
    las_list2, rid_list2, for_list2 = loop_gridsearch(Xdf, ydf, twoyearlists, featurelist, alphas, forest_params, 'twoyear.txt')
    #Three year lists:
    threeyearlists = [[2004, 2005, 2006], [2005, 2006, 2007], [2006, 2007, 2008], [2007, 2008, 2009], [2008, 2009, 2010], [2009, 2010, 2011]]
    las_list3, rid_list3, for_list3 = loop_gridsearch(Xdf, ydf, threeyearlists, featurelist, alphas, forest_params, 'threeyear.txt')
    fouryearlists = [[2003, 2004, 2005, 2006], [2004, 2005, 2006, 2007], [2005, 2006, 2007, 2008], [2006, 2007, 2008, 2009], [2007, 2008, 2009, 2010], [2008, 2009, 2010, 2011]]
    las_list4, rid_list4, for_list4 = loop_gridsearch(Xdf, ydf, fouryearlists, featurelist, alphas, forest_params, 'fouryear.txt')
    fiveyearlists = [[2002, 2003, 2004, 2005, 2006], [2003, 2004, 2005, 2006, 2007], [2004, 2005, 2006, 2007, 2008], [2005, 2006, 2007, 2008, 2009], [2006, 2007, 2008, 2009, 2010], [2007, 2008, 2009, 2010, 2011]]
    las_list5, rid_list5, for_list5 = loop_gridsearch(Xdf, ydf, fiveyearlists, featurelist, alphas, forest_params, 'fiveyear.txt')
