import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
import matplotlib.pyplot as plt
import math
from math import sqrt

def run_regression(model, X, y, alphas):
    grid = GridSearchCV(estimator = model, param_grid = dict(alpha=alphas), cv = 10)
    grid.fit(X, y)
    return grid

def run_regressionlist(model, X, y, alphas, lincv = 10, clu = np.zeros(5)):
    scoring = {'r2': 'r2', 'neg_mean_squared_error': 'neg_mean_squared_error'}
    if(lincv == 'gkf'):
        group = clu
        gkf = list(GroupKFold(n_splits=10).split(X, y, group))
        grid = GridSearchCV(estimator = model, param_grid = dict(alpha=alphas), cv = gkf, scoring = scoring, refit = 'neg_mean_squared_error')
    else:
        grid = GridSearchCV(estimator = model, param_grid = dict(alpha=alphas), cv = lincv, scoring = scoring, refit = 'neg_mean_squared_error')
    grid.fit(X, y)
    return grid

def run_forest(model, X, y, parameters):
    grid = GridSearchCV(model, parameters, cv = 5, verbose = 100)
    grid.fit(X,y)
    return grid

def run_forestlist(model, X, y, parameters, forcv = 5, clu = np.zeros(5)):
    scoring = {'r2': 'r2', 'neg_mean_squared_error': 'neg_mean_squared_error'}
    if(forcv == 'gkf'):
        group = clu
        gkf = list(GroupKFold(n_splits=5).split(X, y, group))
        grid = GridSearchCV(model, parameters, cv = gkf, scoring = scoring, refit = 'neg_mean_squared_error', verbose = 100)
    else:
        grid = GridSearchCV(model, parameters, cv = forcv, scoring = scoring, refit = 'neg_mean_squared_error', verbose = 100)
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

def loop_gridsearch(Xdf, ydf, yearlists, featurelist, alphas, forest_par, filename, lincv = 10, forcv = 5, gap = 1, testsplit = -1, space = False, fiveyear = False):
    #Makes blocks of data from yearlists, and runs through lasso, ridge, and random forest on all of them.  Note that if you're doing k-folds by cluster (gkf), then you must have cluster as a column in your Xdf dataframe.
    f1 = open(filename,"w")
    lasso = Lasso()
    ridge = Ridge()
    forest = RandomForestRegressor()
    las_list = []
    rid_list = []
    for_list = []
    mses = []
    offsets = []
    for i in range(0,len(yearlists)):
        nextyear = [yearlists[i][-1] + gap]
        Xtrainreal = Xdf.copy()
        Xtestreal = Xdf.copy()
        ytrainreal = ydf.copy()
        ytestreal = ydf.copy()
        if(testsplit > -1):
            Xtraining, ytraining, Xtestreal, ytestreal = pick_splits(Xdf, ydf, [testsplit])
            if(space == True):
                Xtrainreal = Xtraining
                ytrainreal = ytraining
        Xt = select_years(Xtrainreal, yearlists[i])
        yt = select_years(ytrainreal, yearlists[i])
        Xt2 = select_years(Xtestreal, nextyear)
        yt2 = select_years(ytestreal, nextyear)
        X = Xt[featurelist]
        targetcol = 'POP_GROWTH_F1'
        if(fiveyear == True):
            targetcol = 'POP_GROWTH_F5'
        y = yt[targetcol]
        Xtest = Xt2[featurelist]
        clu = np.zeros(len(X))
        if('cluster' in list(Xtest)):
            Xtest = Xtest.drop(['cluster'], axis = 1)
            clu = X['cluster'].values
            X = X.drop(['cluster'], axis = 1)
            #print(clu)
        ytest = yt2[targetcol]
        scaler = StandardScaler()
        Xscale = scaler.fit_transform(X)
        Xscaletest = scaler.transform(Xtest)

        #print(len(Xscale), len(y), len(Xtest), len(ytest), len(Xdf), len(ydf))
        #print(len(Xtrainreal), len(ytrainreal), len(X), len(y))


        las_list.append(run_regressionlist(lasso, Xscale, y, alphas, lincv = lincv, clu = clu))
        rid_list.append(run_regressionlist(ridge, Xscale, y, alphas, lincv = lincv, clu = clu))
        for_list.append(run_forestlist(forest, Xscale, y, forest_par, forcv = forcv, clu = clu))
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
        print(nextyear[0], len(yearlists[i]),  las_list[i].cv_results_['mean_test_r2'][las_list[i].best_index_], las_list[i].best_score_,  rid_list[i].cv_results_['mean_test_r2'][rid_list[i].best_index_], rid_list[i].best_score_,  for_list[i].cv_results_['mean_test_r2'][for_list[i].best_index_], for_list[i].best_score_, lasoffset, lasrms, ridoffset, ridrms, foroffset, forrms, mymean, myrms, file = f1)
        print(nextyear[0], len(yearlists[i]), las_list[i].cv_results_['mean_test_r2'][las_list[i].best_index_], las_list[i].best_score_,  rid_list[i].cv_results_['mean_test_r2'][rid_list[i].best_index_], rid_list[i].best_score_,  for_list[i].cv_results_['mean_test_r2'][for_list[i].best_index_], for_list[i].best_score_, lasoffset, lasrms, ridoffset, ridrms, foroffset, forrms, mymean, myrms)
        mses.append((lasrms, ridrms, forrms))
        offsets.append((lasoffset, ridoffset, foroffset))
    las_list.append(yearlists)
    rid_list.append(yearlists)
    for_list.append(yearlists)
    f1.close()
    #Note that I've now changed this so that it returns 5 arguments:
    return las_list, rid_list, for_list, mses, offsets

def loopyears(Xdf, ydf, year, start, stop, featurelist, alphas, forest_par, filename, lincv = 10, forcv = 5):
    # Uses loop_gridsearch to predict the population growth for a given year, using models trained on on the years "year-start" to "year-stop".
    listlists = []
    for i in range(start, stop+1):
        yearlist = []
        for j in range(0, i):
            yearlist.append(year - (i - j))
        listlists.append(yearlist)
    laslist, ridlist, forlist = loop_gridsearch(Xdf, ydf, listlists, featurelist, alphas, forest_par, filename, lincv = lincv, forcv = forcv)
    return laslist, ridlist, forlist

def loopyears_time(Xdf, ydf, depyear1, numyears, listyears, featurelist, alphas, forest_par, filename, lincv = 10, forcv = 5, space = False, fiveyear = False, dosplit = True):
    #Similar to loopyears, but this calculates rmse for a range of "deployment years" for time splits over a fixed number of training years.  E.g., fit separately for deployment in years 2009, 2010, 2011, and 2012.  For 2009, your training data might be the years 2002-2005.  For 2010, they'd be 2003-2006, etc.  Then you average the deployment rmses for each year.
    #I later modified this so that you can do space splits as well.  Just use space = True.
    listlists = []
    for i in range(0, numyears):
        yearlist = []
        thisyear = depyear1 + i
        for j in range(0, len(listyears)):
            yearlist.append(thisyear + listyears[j])
        listlists.append(yearlist)
    if(dosplit == True):
        splitlist = [1, 2, 3, 4, 5]
    else:
        splitlist = [-1]
    #splitlist = [1, 2]
    mselist = []
    offsetlist = []
    for i in range(0, len(splitlist)):
        laslist, ridlist, forlist, mses, offsets = loop_gridsearch(Xdf, ydf, listlists, featurelist, alphas, forest_par, ''.join(['f', str(splitlist[i]), filename]), lincv = lincv, forcv = forcv, gap = -1 * listyears[-1], testsplit = splitlist[i], space = space, fiveyear = fiveyear)
        mselist.append(mses)
        offsetlist.append(offsets)
    flatmselist = [item for sublist in mselist for item in sublist]
    rmsearr = np.zeros(len(flatmselist))
    for i in range(0,len(flatmselist)):
        rmsearr[i] = sqrt(min(flatmselist[i]))
    print(flatmselist, offsetlist, rmsearr)
    print(np.mean(rmsearr))
    return laslist, ridlist, forlist, mses, offsets, rmsearr


def pick_splits(Xdf, ydf, testlist):
    #Removes any splits that are listed in "testlist", because those will be used for test data:
    Xdf2 = Xdf.copy()
    ydf2 = ydf.copy()
    Xdf3 = Xdf2[~Xdf2['split'].isin(testlist)].copy()
    ydf3 = ydf2[~Xdf2['split'].isin(testlist)].copy()
    Xtest = Xdf2[Xdf2['split'].isin(testlist)].copy()
    ytest = ydf2[Xdf2['split'].isin(testlist)].copy()
    #print(5, len(Xdf3), len(ydf3), len(Xtest), len(ytest))
    return Xdf3, ydf3, Xtest, ytest

def apply_model(Xdf, ydf, features, trainyears, testyears, model, target = 'POP_GROWTH_F1'):
    #def apply_model(Xdf, ydf, features, trainyears, testyears, model):
    #Takes a list of years (trainyears), and then does standard scaling on those years in order to get the file into the same format as was used in the fitting.  Then does the same scaling on the "test" data, which is from the same Xdf, but using the "testyears" list.
    Xtrainyr = select_years(Xdf, trainyears)
    Xtrain = Xtrainyr[features].copy()
    Xtestyr = select_years(Xdf, testyears)
    Xtest = Xtestyr[features].copy()
    scaler = StandardScaler()
    Xscale = scaler.fit_transform(Xtrain)
    Xscaletest = scaler.transform(Xtest)
    ypred = model.best_estimator_.predict(Xscaletest)
    yactual = select_years(ydf, testyears)
    yactual['ypredict'] = ypred
    yactual['residual'] = yactual[target] - yactual['ypredict']
    if(target == 'POP_GROWTH_F1'):
        bigtable = Xtest.join(yactual[[target, 'ypredict', 'residual', 'resid_dumb1']])
    else:
        bigtable = Xtest.join(yactual[[target, 'ypredict', 'residual', 'resid_dumb1t5', 'resid_dumb5']])
    return bigtable



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
    #Xdf = pd.read_csv('../DATA/X2000s_nout.csv')
    #ydf = pd.read_csv('../DATA/y2000s_nout.csv')
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
    """oneyearlists = [[2006], [2007], [2008], [2009], [2010], [2011]]
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
    las_list5, rid_list5, for_list5 = loop_gridsearch(Xdf, ydf, fiveyearlists, featurelist, alphas, forest_params, 'fiveyear.txt')"""

    #Try a longer range on the 2009, 2010, 2011, and 2012 data:
    """las_list2012, rid_list2012, for_list2012 = loopyears(Xdf, ydf, 2012, 1, 10, featurelist, alphas, forest_params, 'year2012.txt')
    las_list2011, rid_list2011, for_list2011 = loopyears(Xdf, ydf, 2011, 1, 9, featurelist, alphas, forest_params, 'year2011.txt')
    las_list2010, rid_list2010, for_list2010 = loopyears(Xdf, ydf, 2010, 1, 8, featurelist, alphas, forest_params, 'year2010.txt')
    las_list2009, rid_list2009, for_list2009 = loopyears(Xdf, ydf, 2009, 1, 7, featurelist, alphas, forest_params, 'year2009.txt')"""

    #Here's my code for MVP:
    #Predict 2012 - 2013 population growth using model trained on the previous 5 years:
    """forest_params2 = {'n_estimators':[200], 'max_features':[0.3, 0.5, 0.7], 'min_samples_leaf':[3, 5], 'n_jobs':[-1]}
    las_mvp2012, rid_mvp2012, for_mvp2012 = loopyears(Xdf, ydf, 2012, 10, 10, featurelist, alphas, forest_params2, 'mvpyear2012.txt')

    #Here are some stats on the random forest fit:
    print(for_mvp2012[0].best_score_, for_mvp2012[0].best_params_, for_mvp2012[0].best_estimator_.feature_importances_)"""
    """Outputs this:
    0.464784618544 {'n_estimators': 200, 'max_features': 0.5, 'n_jobs': -1, 'min_samples_leaf': 5} [ 0.20200158  0.35152289  0.01643884  0.01576634  0.014535    0.0124183
    0.01350719  0.01417823  0.01462276  0.01889867  0.01645419  0.0160191
    0.0199875   0.01496215  0.01587672  0.01474533  0.01448669  0.0160427
    0.01636132  0.0294987   0.0260299   0.02672802  0.05109176  0.01254849
    0.01266728  0.02261034]"""

    #And from the Lasso fit:
    #print(las_mvp2012[0].best_score_, las_mvp2012[0].best_params_, las_mvp2012[0].best_estimator_.coef_)
    """0.428498139944 {'alpha': 4.2813323987193961e-06} [  6.79233757e-04   2.89897621e-03  -3.65415817e-05  -2.95148346e-04
   2.66427313e-04  -1.46531893e-04  -1.45809806e-04  -1.16365957e-04
   1.71028896e-04  -3.30219529e-04   1.25247711e-04   1.63371812e-04
   1.46160533e-04  -0.00000000e+00  -2.82630758e-04   2.84978324e-04
  -4.83988841e-05  -5.98574736e-04   2.57896107e-04   7.73185549e-06
  -1.16388906e-04  -3.40103429e-04   1.93702845e-04   9.06024077e-05
   5.85477692e-06   1.63103497e-04]"""

    #Let's plot model predictions vs. actual growth:
    """Xt = select_years(Xdf, [2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011])
    yt = select_years(ydf, [2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011])
    Xt2 = select_years(Xdf, [2012])
    yt2 = select_years(ydf, [2012])
    X = Xt[featurelist]
    y = yt['POP_GROWTH_F1']
    Xtest = Xt2[featurelist]
    ytest = yt2['POP_GROWTH_F1']
    scaler = StandardScaler()
    Xscale = scaler.fit_transform(X)
    Xscaletest = scaler.transform(Xtest)
    ypred = for_mvp2012[0].best_estimator_.predict(Xscaletest)
    plt.scatter(ytest.values, ypred)
    plt.savefig('2012predvsactual.eps')"""

    ################
    ###### Cluster Section
    ################
    #Read in cluster data:
    """Xdf = pd.read_csv('../DATA/X2000s_nout_clu.csv')
    ydf = pd.read_csv('../DATA/y2000s_nout_clu.csv')
    #
    #Split off the test data:
    testlist = [0]
    Xtrain, ytrain, Xtest, ytest = pick_splits(Xdf, ydf, testlist)
    #Select features:
    featurelist = ['POP_GROWTH_P1', 'POP_GROWTH_P2', 'INTPTLAT', 'INTPTLONG', 'age_cohort1', 'age_cohort2', 'age_cohort3', 'age_cohort4', 'age_cohort5', 'age_cohort6', 'age_cohort7', 'age_cohort8', 'age_cohort9', 'age_cohort10', 'age_cohort11', 'age_cohort12', 'age_cohort13', 'age_cohort14', 'age_cohort15', 'age_cohort16', 'age_cohort17', 'age_cohort18', 'logpop', 'logland', 'logwater', 'fem_frac', 'logpop_clu', 'logland_clu', 'pop_growth_clu1', 'pop_growth_clu2']
    alphas = list(np.logspace(-6, 6, num = 20))
    forest_params = {'n_estimators':[200], 'max_features':[0.3, 0.5, 0.7], 'min_samples_leaf':[3, 5], 'n_jobs':[-1]}
    las_mvp2012clu, rid_mvp2012clu, for_mvp2012clu = runmodel.loopyears(Xtrain, ytrain, 2012, 10, 10, featurelist, alphas, forest_params, 'mvpyear2012cluster.txt')

    # Here's a test of gkf:
    forest_paramstest = {'n_estimators':[50], 'max_features':[0.5], 'min_samples_leaf':[3, 5], 'n_jobs':[-1]}
    las_mvp2012test2, rid_mvp2012test2, for_mvp2012test2 = runmodel.loopyears(Xtrain, ytrain, 2012, 2, 2, featurelist, alphas, forest_paramstest, 'mvpyear2012test2.txt')

    featurelist = ['POP_GROWTH_P1', 'POP_GROWTH_P2', 'INTPTLAT', 'INTPTLONG', 'age_cohort1', 'age_cohort2', 'age_cohort3', 'age_cohort4', 'age_cohort5', 'age_cohort6', 'age_cohort7', 'age_cohort8', 'age_cohort9', 'age_cohort10', 'age_cohort11', 'age_cohort12', 'age_cohort13', 'age_cohort14', 'age_cohort15', 'age_cohort16', 'age_cohort17', 'age_cohort18', 'logpop', 'logland', 'logwater', 'fem_frac', 'logpop_clu', 'logland_clu', 'pop_growth_clu1', 'pop_growth_clu2', 'cluster']
    las_mvp2012club, rid_mvp2012club, for_mvp2012club = loopyears(Xtrain, ytrain, 2012, 10, 10, featurelist, alphas, forest_params, 'mvpyear2012clusterb.txt', lincv = 'gkf', forcv = 'gkf')

    #Redo of the cluster section, with new features:
    featurelist = ['POP_GROWTH_P1', 'POP_GROWTH_P2', 'INTPTLAT', 'INTPTLONG', 'age_cohort1', 'age_cohort2', 'age_cohort3', 'age_cohort4', 'age_cohort5', 'age_cohort6', 'age_cohort7', 'age_cohort8', 'age_cohort9', 'age_cohort10', 'age_cohort11', 'age_cohort12', 'age_cohort13', 'age_cohort14', 'age_cohort15', 'age_cohort16', 'age_cohort17', 'age_cohort18', 'logpop', 'logland', 'logwater', 'fem_frac', 'logpop_clu', 'logland_clu', 'pop_growth_clu1', 'pop_growth_clu2',  'growthweight1', 'growthweight2', 'popweight', 'denseweight', 'cluster']
    alphas = list(np.logspace(-6, 6, num = 20))
    forest_params = {'n_estimators':[200], 'max_features':[0.3, 0.5, 0.7], 'min_samples_leaf':[3, 5], 'n_jobs':[-1]}
    las_mvp2012test, rid_mvp2012test, for_mvp2012test = runmodel.loopyears(Xtrain, ytrain, 2012, 10, 10, featurelist, alphas, forest_params, 'mvpyear2012clustertest.txt', lincv = 'gkf', forcv = 'gkf')"""

    #################################################
    #Trying again with cluster analysis, new features:
    #################################################
    Xdf = pd.read_csv('../DATA/X2000s_nout_clug.csv')
    ydf = pd.read_csv('../DATA/y2000s_nout_clug.csv')
    #
    #Split off the test data:
    testlist = [0]
    Xtrain, ytrain, Xtest, ytest = pick_splits(Xdf, ydf, testlist)
    #Select features:
    featurelist = ['POP_GROWTH_P1', 'POP_GROWTH_P2', 'INTPTLAT', 'INTPTLONG', 'age_cohort1', 'age_cohort2', 'age_cohort3', 'age_cohort4', 'age_cohort5', 'age_cohort6', 'age_cohort7', 'age_cohort8', 'age_cohort9', 'age_cohort10', 'age_cohort11', 'age_cohort12', 'age_cohort13', 'age_cohort14', 'age_cohort15', 'age_cohort16', 'age_cohort17', 'age_cohort18', 'logpop', 'logland', 'logwater', 'fem_frac', 'growthweight120', 'growthweight220', 'growthweight320', 'growthweight420', 'growthweight520', 'popweight20', 'denseweight20', 'growthweight150', 'growthweight250', 'growthweight350', 'growthweight450', 'growthweight550', 'popweight50', 'denseweight50', 'growthweight1100', 'growthweight2100', 'growthweight3100', 'growthweight4100', 'growthweight5100', 'popweight100', 'denseweight100', 'growthweight1200', 'growthweight2200', 'growthweight3200', 'growthweight4200', 'growthweight5200', 'popweight200', 'denseweight200', 'cluster']
    featuretemp = ['POP_GROWTH_P1', 'POP_GROWTH_P2', 'INTPTLAT', 'INTPTLONG', 'age_cohort1', 'age_cohort2', 'age_cohort3', 'age_cohort4', 'age_cohort5', 'age_cohort6', 'age_cohort7', 'age_cohort8', 'age_cohort9', 'age_cohort10', 'age_cohort11', 'age_cohort12', 'age_cohort13', 'age_cohort14', 'age_cohort15', 'age_cohort16', 'age_cohort17', 'age_cohort18', 'logpop', 'logland', 'logwater', 'fem_frac', 'cluster']
    alphas = list(np.logspace(-6, 6, num = 20))
    forest_params = {'n_estimators':[50], 'max_features':[0.3, 0.5, 0.7], 'min_samples_leaf':[5], 'n_jobs':[-1]}
    #forest_paramstest = {'n_estimators':[100], 'max_features':[0.5], 'min_samples_leaf':[3, 5], 'n_jobs':[-1]}
    las_mvp2012clug, rid_mvp2012clug, for_mvp2012clug = loopyears(Xtrain, ytrain, 2012, 10, 10, featurelist, alphas, forest_params, 'mvpyear2012clugaus.txt', lincv = 'gkf', forcv = 'gkf')
    # mse = -1.85483172154e-05 (random forest)
    # Try it without any of the environmental parameters:
    las_mvp2012clug5, rid_mvp2012clug5, for_mvp2012clug5 = loopyears(Xtrain, ytrain, 2012, 10, 10, featuretemp, alphas, forest_params, 'mvpyear2012clugaust.txt', lincv = 'gkf', forcv = 'gkf')
    # Without the environmental parameters, mse = -1.89042296212e-05
    # Try a very dumb model with only pop growth of past two years:
    featuretemp2 = ['POP_GROWTH_P1', 'POP_GROWTH_P2', 'cluster']
    las_mvp2012clugc, rid_mvp2012clugc, for_mvp2012clugc = loopyears(Xtrain, ytrain, 2012, 10, 10, featuretemp2, alphas, forest_params, 'mvpyear2012clugausc.txt', lincv = 'gkf', forcv = 'gkf')
    #mse = -2.09819277391e-05
    #The environmental parameters help a little, but not by much.

    #OK now I've added loopyears_time, and am going to test it out:
    featuretemp2 = ['POP_GROWTH_P1', 'POP_GROWTH_P2', 'INTPTLAT', 'INTPTLONG', 'age_cohort1', 'age_cohort2', 'age_cohort3', 'age_cohort4', 'age_cohort5', 'age_cohort6', 'age_cohort7', 'age_cohort8', 'age_cohort9', 'age_cohort10', 'age_cohort11', 'age_cohort12', 'age_cohort13', 'age_cohort14', 'age_cohort15', 'age_cohort16', 'age_cohort17', 'age_cohort18', 'logpop', 'logland', 'logwater', 'fem_frac']
    las_test, rid_test, for_test, mses, offsets, rmsearr = loopyears_time(Xtrain, ytrain, 2009, 2, [-4, -3], featuretemp2, alphas, forest_params, 'testfile.txt')

    #OK, that seemed to work.  Now I'll try fitting four year data (right after I commit).
    featuretime = ['POP_GROWTH_P1', 'POP_GROWTH_P2', 'INTPTLAT', 'INTPTLONG', 'age_cohort1', 'age_cohort2', 'age_cohort3', 'age_cohort4', 'age_cohort5', 'age_cohort6', 'age_cohort7', 'age_cohort8', 'age_cohort9', 'age_cohort10', 'age_cohort11', 'age_cohort12', 'age_cohort13', 'age_cohort14', 'age_cohort15', 'age_cohort16', 'age_cohort17', 'age_cohort18', 'logpop', 'logland', 'logwater', 'fem_frac']
    forest_params = {'n_estimators':[100], 'max_features':[0.3, 0.5, 0.7], 'min_samples_leaf':[5], 'n_jobs':[-1]}
    las_4year, rid_4year, for_4year, mse4, offset4, rmsearr4 = loopyears_time(Xtrain, ytrain, 2009, 4, [-7, -6, -5, -4], featuretime, alphas, forest_params, '_4yearfile.txt')
    #Result is 0.0046149641105588113

    #And then yere is is 3 years:
    las_3year, rid_3year, for_3year, mse3, offset3, rmsearr3 = loopyears_time(Xtrain, ytrain, 2009, 4, [-6, -5, -4], featuretime, alphas, forest_params, '_3yearfile.txt')
    #Result is 0.00463592608369

    #Here is 2 years:
    las_2year, rid_2year, for_2year, mse2, offset2, rmsearr2 = loopyears_time(Xtrain, ytrain, 2009, 4, [-5, -4], featuretime, alphas, forest_params, '_2yearfile.txt')
    #Result is 0.0046456275339

    #Here is 1 year:
    las_1year, rid_1year, for_1year, mse1, offset1, rmsearr1 = loopyears_time(Xtrain, ytrain, 2009, 4, [-4], featuretime, alphas, forest_params, '_1yearfile.txt')
    #Result is 0.00460483626665

    #Need to copy over the updated version of loopyears_time from scratchpaper.py, and put it in this file.
    #Now, move on to splitting in space.....I'll try 4 years first:
    featurespace = ['POP_GROWTH_P1', 'POP_GROWTH_P2', 'INTPTLAT', 'INTPTLONG', 'age_cohort1', 'age_cohort2', 'age_cohort3', 'age_cohort4', 'age_cohort5', 'age_cohort6', 'age_cohort7', 'age_cohort8', 'age_cohort9', 'age_cohort10', 'age_cohort11', 'age_cohort12', 'age_cohort13', 'age_cohort14', 'age_cohort15', 'age_cohort16', 'age_cohort17', 'age_cohort18', 'logpop', 'logland', 'logwater', 'fem_frac', 'growthweight120', 'growthweight220',  'popweight20', 'denseweight20', 'growthweight150', 'growthweight250', 'popweight50', 'denseweight50', 'growthweight1100', 'growthweight2100',  'popweight100', 'denseweight100', 'growthweight1200', 'growthweight2200',  'popweight200', 'denseweight200', 'cluster']
    las_4year_s, rid_4year_s, for_4year_s, mse4_s, offset4_s, rmsearr4_s = loopyears_time(Xtrain, ytrain, 2009, 4, [-4, -3, -2, -1], featurespace, alphas, forest_params, '_4yearfile_s.txt', lincv = 'gkf', forcv = 'gkf', space = True)
    #Result is 0.00432388284704

    #And now 5 years:
    las_5year_s, rid_5year_s, for_5year_s, mse5_s, offset5_s, rmsearr5_s = loopyears_time(Xtrain, ytrain, 2009, 4, [-5, -4, -3, -2, -1], featurespace, alphas, forest_params, '_5yearfile_s.txt', lincv = 'gkf', forcv = 'gkf', space = True)
    #Result is 0.0043314548059

    #And now 6 years:
    las_6year_s, rid_6year_s, for_6year_s, mse6_s, offset6_s, rmsearr6_s = loopyears_time(Xtrain, ytrain, 2009, 4, [-6, -5, -4, -3, -2, -1], featurespace, alphas, forest_params, '_6yearfile_s.txt', lincv = 'gkf', forcv = 'gkf', space = True)
    #Result is 0.00432316291375

    #And now 7 years:
    las_7year_s, rid_7year_s, for_7year_s, mse7_s, offset7_s, rmsearr7_s = loopyears_time(Xtrain, ytrain, 2009, 4, [-7, -6, -5, -4, -3, -2, -1], featurespace, alphas, forest_params, '_7yearfile_s.txt', lincv = 'gkf', forcv = 'gkf', space = True)
    #Result is 0.00432769513015

    #Six year spatial fit is the best.

    #Here's a test of doing group k-folds even when train/test split is done in time:
    featuretimegroup = ['POP_GROWTH_P1', 'POP_GROWTH_P2', 'INTPTLAT', 'INTPTLONG', 'age_cohort1', 'age_cohort2', 'age_cohort3', 'age_cohort4', 'age_cohort5', 'age_cohort6', 'age_cohort7', 'age_cohort8', 'age_cohort9', 'age_cohort10', 'age_cohort11', 'age_cohort12', 'age_cohort13', 'age_cohort14', 'age_cohort15', 'age_cohort16', 'age_cohort17', 'age_cohort18', 'logpop', 'logland', 'logwater', 'fem_frac', 'cluster']
    #Here is 1 year:
    las_1yeargroup, rid_1yeargroup, for_1yeargroup, mse1group, offset1group, rmsearr1group = loopyears_time(Xtrain, ytrain, 2009, 4, [-4], featuretimegroup, alphas, forest_params, '_1yearfilegroup.txt', lincv = 'gkf', forcv = 'gkf')
    #Answer is 0.00463037088313.  So, slightly worse than not doing the group k-fold, and fitting everything else the same.

    #Now try fitting in space, but without the environmental parameters:
    #6 years:
    las_6year_sno, rid_6year_sno, for_6year_sno, mse6_sno, offset6_sno, rmsearr6_sno = loopyears_time(Xtrain, ytrain, 2009, 4, [-6, -5, -4, -3, -2, -1], featuretimegroup, alphas, forest_params, '_6yearfile_sno.txt', lincv = 'gkf', forcv = 'gkf', space = True)
    #Result is 0.00442128324521

    ############################################
    ###Fit model just on counties with population > 25,000
    ############################################
    #Pick counties with population greater than 25,000:
    Xtrainhigh = Xtrain[Xtrain['logpop'] > 4.39793].copy()
    ytrainhigh = ytrain[Xtrain['logpop'] > 4.39793].copy()
    las_6year_sh, rid_6year_sh, for_6year_sh, mse6_sh, offset6_sh, rmsearr6_sh = loopyears_time(Xtrainhigh, ytrainhigh, 2009, 4, [-6, -5, -4, -3, -2, -1], featurespace, alphas, forest_params, '_6yearfile_sh.txt', lincv = 'gkf', forcv = 'gkf', space = True, dosplit = False)
    #Result is 0.00285578979282

    #Output a sample model:
    trainyears = [2003, 2004, 2005, 2006, 2007, 2008]
    testyears = [2009]
    fortable_sh = apply_model(Xtrainhigh, ytrainhigh, featurespace[:-1], trainyears, testyears, for_6year_sh[0])
    fortable_sh.to_csv('../DATA/fortable_sh.csv')

    ############################################
    ###Five year prediction fits
    ############################################
    Xdf5 = pd.read_csv('../DATA/X2000s_nout_clug5.csv')
    ydf5 = pd.read_csv('../DATA/y2000s_nout_clug5.csv')
    #Split off the test data:
    testlist = [0]
    Xtrain5, ytrain5, Xtest5, ytest5 = pick_splits(Xdf5, ydf5, testlist)

    las_4year_s5, rid_4year_s5, for_4year_s5, mse4_s5, offset4_s5, rmsearr4_s5 = loopyears_time(Xtrain5, ytrain5, 2007, 4, [-4, -3, -2, -1], featurespace, alphas, forest_params, '_4yearfile_s5.txt', lincv = 'gkf', forcv = 'gkf', space = True, fiveyear = True, dosplit = False)
    #Result is 0.01389807

    #And now 5 years:
    las_5year_s5, rid_5year_s5, for_5year_s5, mse5_s5, offset5_s5, rmsearr5_s5 = loopyears_time(Xtrain5, ytrain5, 2007, 4, [-5, -4, -3, -2, -1], featurespace, alphas, forest_params, '_5yearfile_s5.txt', lincv = 'gkf', forcv = 'gkf', space = True, fiveyear = True, dosplit = False)
    #Result is 0.0133544983852

    #Output a 5 year model:
    trainyears = [2004, 2005, 2006, 2007, 2008]
    testyears = [2009]
    fortable_s5 = apply_model(Xtrain5, ytrain5, featurespace[:-1], trainyears, testyears, for_5year_s5[2], target = 'POP_GROWTH_F5')
    fortable_s5.to_csv('../DATA/fortable_s5.csv')
