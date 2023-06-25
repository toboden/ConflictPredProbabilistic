import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import os
from scipy.stats import nbinom
from scipy.stats import poisson
import CRPS.CRPS as pscore

# create the feature- and actuals-data list
# set the feature and actuals year lists
feature_years = ['2017','2018','2019','2020']
actual_years = ['2018','2019','2020','2021']

actuals_df_list = []
features_df_list = []

# path to the current directory
current_dir = os.getcwd()

for i in range(len(feature_years)):
    # relative paths to the parquet files
    relative_path_features = os.path.join('..', 'data', 'cm_features_to_oct' + feature_years[i] + '.parquet')
    relative_path_actuals = os.path.join('..', 'data', 'cm_actuals_' + actual_years[i] + '.parquet')

    path_features = os.path.join(current_dir, relative_path_features)
    path_actuals = os.path.join(current_dir, relative_path_actuals)

    # append datasets to the lists
    actuals_df_list.append({'year':actual_years[i], 'data':pd.read_parquet(path_actuals, engine='pyarrow')})
    features_df_list.append({'year':feature_years[i], 'data':pd.read_parquet(path_features, engine='pyarrow')})

# concat the feature datasets, so that every data contains the observations starting with january 1990
for i in range(1,len(features_df_list)):
    features_df_list[i]['data'] = pd.concat([features_df_list[i-1]['data'], features_df_list[i]['data']])

# function to check, if the last n months are in the dataset of a country,
# other than that the last month of a country in the feature dataset has to be 3 months before the first actuals month!!
def check_last_nMonths(n, country_id, yearindex):
    country = country_feature_group_list[yearindex].get_group(country_id)

    # reference month of the actual dataset
    actual_month_list = actuals_df_list[yearindex]['data'].index.get_level_values('month_id').unique().tolist()

    # if the last month of the feature dataset in the country does not match the first of the actuals return false
    if (actual_month_list[0] - 3) != country.index.get_level_values('month_id').unique().tolist()[-1]:
        return False
    else:
        month_list = features_df_list[yearindex]['data'].index.get_level_values('month_id').unique().tolist()
        last_month = month_list[-1] # equals the first month - 3 from the corresponding actuals dataset
        first_month = month_list[0]

        last_n_months = True

        if last_month-n+1 < first_month:
            last_n_months = False
        else:
            month_list = list(range(last_month-n+1, last_month+1))
            
            for month in month_list:
                if month not in country.index.get_level_values('month_id'):
                    last_n_months = False
                    break

        return last_n_months
        #return True

def nBinom_quantiles(featureSeries, w, quantiles):
        if w == 'None':
             # calculate n (r) and p via average/variance
            mean = pd.Series.mean(featureSeries)
            var = pd.Series.var(featureSeries)
        else:
            # calculate n (r) and p via average/variance
            mean = pd.Series.mean(featureSeries.tail(w).loc[:,'ged_sb'])
            var = pd.Series.var(featureSeries.tail(w).loc[:,'ged_sb'])

        #hier verteilung = nbinom ppf als
        dummy_fatalities_list = []

        # string to store distribution
        dist_string = ''

        if var != 0 and var > mean:
                n = (mean**2) / (var - mean) # equivalent to r
                p = mean / var
                dummy_fatalities_list = nbinom.ppf(quantiles, n, p).tolist()

                dist_string = 'NBinom'

        elif var != 0 and var <= mean:
                dummy_fatalities_list = poisson.ppf(quantiles, mean).tolist()

                dist_string = 'Pois'

        else:
                dummy_fatalities_list = [0] * 999
                dist_string = 'None'

        return {'fatalities': dummy_fatalities_list, 'dist': dist_string, 'mean': mean, 'var': var}

#--------------------------------------------------------------------------------------------
# because of the concatination only the last dataframe is used (later on the appended months are dropped for datasets before 2020)

# IF THIS CODE IS USED FOR THE 2024 PREDICTION ADJUST THE features_1990to2020 in the whole file to 1990to2023

features_1990to2020_df = features_df_list[3]['data']
country_list = sorted(features_df_list[3]['data'].index.get_level_values('country_id').unique().tolist())

# country group list of all four datasets
country_feature_group_list = []
country_actual_group_list = []
# fill list 
for i in range(len(features_df_list)):
    country_feature_group_list.append(features_df_list[i]['data'].groupby('country_id'))
    country_actual_group_list.append(actuals_df_list[i]['data'].groupby('country_id'))

""" # same reason as mentioned two lines earlier
country_feature_group_1990to2020 = country_feature_group_list[3] """


print(len(country_list))


# modify country_list so that it contains only country_ids 
# that have at least the last n months of observations in the last dataset (2020)!
numberMonths_toOct20 = 96 # 96 = 5*12 (5 jahre für 2017) + 3*12 (jedes Jahr 12 Monate mehr also 2020 8 Jahre)
#ABER: um konsistent zu bleiben wird für jedes Jahr (jeden to_octX Datensatz) nur die letzten 5 Jahre verwendet!!!

#-- note------
# dataset 2020 is used, because of the structure of the other datasets.
# 2020 is dataset 2019 with 12 additional rows (months) etc.
# for the CRPS calculation  of the datasets != 2020 the last 12*x windows are deleted
# this procedure is saving computation time
#-------------


#IMPORTANT
#if you do not minimize over all countries but only the single countries, 
# it is sufficient to check if all countries contain the last month in the features dataset (this way you use the full information). 
# But you still have to check check_last_nMonths(len(countrymonths), countryIndex, 3), so that no month is missing in between.

# => so currently not all information is used for each country

dummy_list = []
for countryIndex in country_list:
    dummy_hasLastN_months = True

    # index 3 is the last dataset
    # 76, da Land 246 z.b. genau die letzten 112 Monate (in '2020') als Beobachtungen hat 
    if check_last_nMonths(numberMonths_toOct20, countryIndex, 3) is not True:
        dummy_hasLastN_months = False  
    
    if dummy_hasLastN_months is True:
        dummy_list.append(countryIndex)

# the values in country_list are the 'country_id'
country_list = dummy_list

#IMPORTANT
# all countries that have the last month as observation have the last 96 months as observations (in 2020)!!! so no country is excluded
# checked by modifing the check_last_nMonths function -> else: return True

print(len(country_list))

# list of the (prediction) windows
window_list = list(range(2, 25))
s_prediction_list = list(range(3, 15))

## changes, so that the calculation does not take a long time -------------------
#shorter windows
#window_list = list(range(2, 13))
# remove all but ten countries
elements_to_remove = country_list[0:(len(country_list)-4)]
#country_list = [element for element in country_list if element not in elements_to_remove]
print(len(country_list))


number_countries = len(country_list)
number_dataframes = len(actual_years)
number_w = len(window_list)

# lists for the estimation of the distribution
quantiles = np.arange(0.001, 0.9999, 0.001)
quantiles = [round(q, 3) for q in quantiles] # due to binary inaccuracies
string_quantile_list = [f"{round(q * 100, 1)}%" for q in quantiles] # sting values of the quantiles

# last month of the dataframe as reference for the moving prediction windows
last_month = features_df_list[3]['data'].index.get_level_values('month_id').tolist()[-1]

# list to store the estimations/predictions for each w
baseline_estimate_list = []

# loop through windows
for i in range(number_w):    

    print('                              window ' + str(i+1) + '/' + str(number_w)  , end='\r')

    w = window_list[i] # current window
    baseline_estimate_list.append({'window':w, 
                                'country_predict_list':[{'country_id':country, 'predictionWindowsN':[]} for country in country_list]})
    
    #calculate the number of subsets, that are used to estimate the distribution and validate it via 12 months of actuals 
    # the number is dependent of the actual w. E.g. with the maximal w (e.g. 24): if w=24, actuals are 12 months (starting with s=3 to s=14) 
    # -> 24 + 2 + 12 = 39 observations of ged_sb per window
    # so if the dataset has 96 observations there are 96 - 38 = 58 shiftable windows for 2020
    numberWindows = numberMonths_toOct20 - (w + 2 + 12)

    windowLength = w + 2 + 12 # length of the individual window for the current w
    
    # loop through all countries
    for index in range(number_countries):
        country = country_list[index]
    
        print('country ' + str(index+1) + '/' + str(number_countries), end='\r')

        features = country_feature_group_list[3].get_group(country) # features of country
        

        # loop through all X equal parts of the feature dataset (traindata length w, actuals is vector of the next t+3 till t+12 observations)
        for j in range(numberWindows):
            starting_month_window = last_month - windowLength + 1 - numberWindows + 1  + j
            ending_month_window = starting_month_window + w - 1

            starting_month_actuals = ending_month_window + 3
            ending_month_actuals = starting_month_actuals + 11
            
            window_features = features.loc[(slice(starting_month_window, ending_month_window), slice(None)), 'ged_sb']
            window_actuals = features.loc[(slice(starting_month_actuals, ending_month_actuals), slice(None)), 'ged_sb']

            predict = nBinom_quantiles(window_features, 'None', quantiles)
            
            baseline_estimate_list[i]['country_predict_list'][index]['predictionWindowsN'].append(
                [{'country_id':country, 'w':w, 'dist':predict['dist'], 
                'mean':predict['mean'], 'var':predict['var'], 'first_month_feature':starting_month_window, 
                'quantile':string_quantile_list, 'fatalities':predict['fatalities']}, 
                {'s':s_prediction_list, 
                    'month_id': window_actuals.index.get_level_values('month_id'),
                    'unreal_actuals':window_actuals.values}])

# with 10 countries 2m





import copy
# lists to store all crps values
baseline_crps_list_to_oct20 = [
    {
        'country_id': country,
        'baseline': [
            {'s': s, 'w': [], 'CRPS': []}
            for s in s_prediction_list
        ]
    }
    for country in country_list
]
baseline_crps_list_to_oct19 = copy.deepcopy(baseline_crps_list_to_oct20)
baseline_crps_list_to_oct18 = copy.deepcopy(baseline_crps_list_to_oct20)
baseline_crps_list_to_oct17 = copy.deepcopy(baseline_crps_list_to_oct20)

# number of prediction windows
number_s = len(s_prediction_list)

# fill lists with crps calculations
for s in s_prediction_list:
    print('                  prediction window ' + str(s-2) + '/' + str(number_s), end='\r')

    for index in range(number_countries):
        country = country_list[index]
        print('country ' + str(index+1) + '/' + str(number_countries), end='\r')
            
        for i in range(number_w):
            w = window_list[i]
            dummy_crps_list = [] 

            # loop over all subset windows of the country and w 
            for j in range(len(baseline_estimate_list[i]['country_predict_list'][index]['predictionWindowsN'])):

                distribution = baseline_estimate_list[i]['country_predict_list'][index]['predictionWindowsN'][j][0]['fatalities']
                actual = baseline_estimate_list[i]['country_predict_list'][index]['predictionWindowsN'][j][1]['unreal_actuals'][s-3]

                crps = pscore(np.array(distribution),actual).compute()[0]
                dummy_crps_list.append(crps)

            # dataframe to_oct17
            baseline_crps_list_to_oct17[index]['baseline'][s-3]['w'].append(w)
            baseline_crps_list_to_oct17[index]['baseline'][s-3]['CRPS'].append(np.mean(dummy_crps_list[:-(3*12)]))

            # dataframe to_oct18
            baseline_crps_list_to_oct18[index]['baseline'][s-3]['w'].append(w)
            baseline_crps_list_to_oct18[index]['baseline'][s-3]['CRPS'].append(np.mean(dummy_crps_list[12:-(2*12)]))

            # dataframe to_oct19
            baseline_crps_list_to_oct19[index]['baseline'][s-3]['w'].append(w)
            baseline_crps_list_to_oct19[index]['baseline'][s-3]['CRPS'].append(np.mean(dummy_crps_list[(2*12):-12]))

            # dataframe to_oct20
            baseline_crps_list_to_oct20[index]['baseline'][s-3]['w'].append(w)
            baseline_crps_list_to_oct20[index]['baseline'][s-3]['CRPS'].append(np.mean(dummy_crps_list[(3*12):]))


task2_baseline_list = [baseline_crps_list_to_oct17, baseline_crps_list_to_oct18,
                       baseline_crps_list_to_oct19, baseline_crps_list_to_oct20]

# with 10 countries 18m




# list to store the results of the minimal w's
w_minimization_list = [{'predictionYear':year, 'minWData':[]} for year in actual_years]

# list to store the list to compute the minimal w's
w_compute_list = [{'predictionYear':year, 'data':[]} for year in actual_years]

# loop over the four different datasets to predict (18-21)
for task2_index in range(len(task2_baseline_list)):
    v1_baseline_crps_dict = {'w':[],'CRPS':[]}
    v2_baseline_crps_list = [{'country_id': country, 'baseline': {'w':[],'CRPS':[]}} for country in country_list]
    v3_baseline_crps_list = [{'s':s,'w':[],'CRPS':[]} for s in s_prediction_list]

    ## baseline v1---------------------------------------------------------------------------
    # loop over w
    for j in range(number_w):
        w = window_list[j]
        dummy_crps_v1_list = []
        # loop over countries
        for i in range(number_countries):
            # loop over prediction windows s
            for k in range(number_s):
                dummy_crps_v1_list.append(task2_baseline_list[task2_index][i]['baseline'][k]['CRPS'][j])
        v1_baseline_crps_dict['w'].append(w)
        v1_baseline_crps_dict['CRPS'].append(np.mean(dummy_crps_v1_list))

    v1_baseline_crps = pd.DataFrame(v1_baseline_crps_dict)

    w_compute_list[task2_index]['data'].append(v1_baseline_crps)

    v1_baseline_crps = v1_baseline_crps[v1_baseline_crps.CRPS == v1_baseline_crps.loc[:,'CRPS'].min()]
    v1_baseline_crps.set_index(pd.Index(range(len(v1_baseline_crps))), inplace=True)
        
    w_minimization_list[task2_index]['minWData'].append(v1_baseline_crps)
    #----------------------------------------------------------------------------------------

    ## baseline v2----------------------------------------------------------------------------
    # list for baseline v2
    for i in range(number_countries):
        for j in range(number_w):
            w = window_list[j]
            dummy_crps_v2_list = []
            for k in range(number_s):
                dummy_crps_v2_list.append(task2_baseline_list[task2_index][i]['baseline'][k]['CRPS'][j])
            v2_baseline_crps_list[i]['baseline']['w'].append(w)
            v2_baseline_crps_list[i]['baseline']['CRPS'].append(np.mean(dummy_crps_v2_list))
        
    # dataframe with the w that minimizes the CRPS for every country (v2)
    data_v2 = {
        'country_id':[],
        'w':[],
        'CRPS':[]
    }
    for i in range(len(v2_baseline_crps_list)):
        # get the index of the minimal CRPS value
        min_index = v2_baseline_crps_list[i]['baseline']['CRPS'].index(min(v2_baseline_crps_list[i]['baseline']['CRPS']))
        
        # store values in dict
        data_v2['country_id'].append(v2_baseline_crps_list[i]['country_id'])
        data_v2['w'].append(v2_baseline_crps_list[i]['baseline']['w'][min_index])
        data_v2['CRPS'].append(v2_baseline_crps_list[i]['baseline']['CRPS'][min_index])
        
    v2_baseline_crps = pd.DataFrame(data_v2)
    w_minimization_list[task2_index]['minWData'].append(v2_baseline_crps)
    w_compute_list[task2_index]['data'].append(v2_baseline_crps_list)
    #----------------------------------------------------------------------------------------


    ## baseline v3---------------------------------------------------------------------------
    for s_index in range(number_s):
        dummy_crps_v3_list = []
        s = s_prediction_list[s_index]
        for w_index in range(number_w):
            w = window_list[w_index]
            for i in range(number_countries):
                dummy_crps_v3_list.append(task2_baseline_list[task2_index][i]['baseline'][s_index]['CRPS'][w_index])
            v3_baseline_crps_list[s_index]['w'].append(w)
            v3_baseline_crps_list[s_index]['CRPS'].append(np.mean(dummy_crps_v3_list))

    # dataframe with the w that minimize the CRPS for each prediction window s
    data_v3 = {
        's':[],
        'w':[],
        'CRPS':[]
    }
    # length of the v3_baseline list is the number of prediction windows
    for i in range(len(v3_baseline_crps_list)):
        s = s_prediction_list[i]
        # get the index of the minimal CRPS value
        min_index = v3_baseline_crps_list[i]['CRPS'].index(min(v3_baseline_crps_list[i]['CRPS']))

        # store values in dict
        data_v3['s'].append(s)
        data_v3['w'].append(v3_baseline_crps_list[i]['w'][min_index])
        data_v3['CRPS'].append(v3_baseline_crps_list[i]['CRPS'][min_index])

    v3_baseline_crps = pd.DataFrame(data_v3)

    w_minimization_list[task2_index]['minWData'].append(v3_baseline_crps)
    w_compute_list[task2_index]['data'].append(v3_baseline_crps_list)
    #----------------------------------------------------------------------------------------

    ## baseline v4---------------------------------------------------------------------------
    v4_baseline_crps = [{'country_id':country,
                        's':[],
                        'w':[],
                        'CRPS':[]
                        } for country in country_list]

    # loop over all countries
    for i in range(len(task2_baseline_list[task2_index])):
        # loop over all prediction windows
        for s_index in range(number_s):
            s = s_prediction_list[s_index]
            # get the index of the minimal CRPS value
            min_index = task2_baseline_list[task2_index][i]['baseline'][s_index]['CRPS'].index(min(task2_baseline_list[task2_index][i]['baseline'][s_index]['CRPS']))
        
            # store values in dict
            v4_baseline_crps[i]['s'].append(s)
            v4_baseline_crps[i]['w'].append(task2_baseline_list[task2_index][i]['baseline'][s_index]['w'][min_index])
            v4_baseline_crps[i]['CRPS'].append(task2_baseline_list[task2_index][i]['baseline'][s_index]['CRPS'][min_index])

        v4_baseline_crps[i] = pd.DataFrame(v4_baseline_crps[i])

    w_minimization_list[task2_index]['minWData'].append(v4_baseline_crps)
    w_compute_list[task2_index]['data'].append(task2_baseline_list[task2_index])
    #----------------------------------------------------------------------------------------






    # list to save the predictions for each country
baseline_prediction_list = [[{'country_id': country, 'base': 1, 'prediction': {'2018': [], 
                                                                               '2019': [], 
                                                                               '2020': [], 
                                                                               '2021': []}} for country in country_list],
                            [{'country_id': country, 'base': 2, 'prediction': {'2018': [], 
                                                                               '2019': [], 
                                                                               '2020': [], 
                                                                               '2021': []}} for country in country_list],
                            [{'country_id': country, 'base': 3, 'prediction': {'2018': [], 
                                                                               '2019': [], 
                                                                               '2020': [], 
                                                                               '2021': []}} for country in country_list],
                            [{'country_id': country, 'base': 4, 'prediction': {'2018': [], 
                                                                               '2019': [], 
                                                                               '2020': [], 
                                                                               '2021': []}} for country in country_list]]

quantiles = np.arange(0.001, 0.9999, 0.001)
quantiles = [round(q, 3) for q in quantiles] # due to binary inaccuracies
string_quantile_list = [f"{round(q * 100, 1)}%" for q in quantiles]


# loop through all countries (that are present in each dataset)
for index in range(number_countries):
    country = country_list[index]

    # list to store the predictions for each year temporally
    baseline_prediction = [[] for _ in range(number_dataframes)]
    
    # loop through datasets
    for i in range(number_dataframes): 
        features = country_feature_group_list[i].get_group(country) # features of country in dataset i
        actuals = country_actual_group_list[i].get_group(country) # actuals of country in dataset i

        data_year = actual_years[i]

        # loop over the four different baseline minimization methods
        for j in range(len(w_minimization_list[i]['minWData'])):

            # baseline 1
            if j == 0:
                w = w_minimization_list[i]['minWData'][j].loc[0,'w'] # use the w obtained by minimization on the feature dataset
                fit = nBinom_quantiles(features, w, quantiles) # calculate the quantiles for the w

                dummy_crps_list = []
                for s in s_prediction_list:
                    true_obs = actuals.iloc[s-3,0] # true observation of the month s
                    NB_prediction = fit['fatalities'] # value of the quantiles
                    crps = pscore(np.array(NB_prediction),true_obs).compute()[0] # compute crps
                    dummy_crps_list.append(crps)

                baseline_prediction_list[j][index]['prediction'][data_year].append({'s':s_prediction_list, 'w':w, 
                                                                                    'dist':fit['dist'], 'mean':fit['mean'],
                                                                                    'var':fit['var'], 
                                                                                    'quantile':string_quantile_list, 
                                                                                    'fatalities':fit['fatalities'],
                                                                                    'actual':actuals.iloc[:,0].tolist(),
                                                                                    'CRPS':dummy_crps_list})

            # baseline 2
            elif j == 1:
                if country == w_minimization_list[i]['minWData'][j].loc[index,'country_id']:
                    w = w_minimization_list[i]['minWData'][j].loc[index,'w']
                    fit = nBinom_quantiles(features, w, quantiles) # calculate the quantiles for the w

                    dummy_crps_list = []
                    for s in s_prediction_list:
                        true_obs = actuals.iloc[s-3,0] # true observation of the month s
                        NB_prediction = fit['fatalities'] # value of the quantiles
                        crps = pscore(np.array(NB_prediction),true_obs).compute()[0] # compute crps
                        dummy_crps_list.append(crps)

                    baseline_prediction_list[j][index]['prediction'][data_year].append({'s':s_prediction_list, 'w':w, 
                                                                                        'dist':fit['dist'], 'mean':fit['mean'],
                                                                                        'var':fit['var'], 
                                                                                        'quantile':string_quantile_list, 
                                                                                        'fatalities':fit['fatalities'],
                                                                                        'actual':actuals.iloc[:,0].tolist(),
                                                                                        'CRPS':dummy_crps_list})
                else:
                    print('Stopp')
                    break

            # baseline 3
            elif j == 2:
                for s in s_prediction_list:
                    w = w_minimization_list[i]['minWData'][j].loc[s-3,'w']
                    fit = nBinom_quantiles(features, w, quantiles) # calculate the quantiles for the w

                    true_obs = actuals.iloc[s-3,0] # true observation of the month s
                    NB_prediction = fit['fatalities'] # value of the quantiles
                    crps = pscore(np.array(NB_prediction),true_obs).compute()[0] # compute crps

                    baseline_prediction_list[j][index]['prediction'][data_year].append({'s':s, 'w':w, 
                                                                                        'dist':fit['dist'], 'mean':fit['mean'],
                                                                                        'var':fit['var'], 
                                                                                        'quantile':string_quantile_list, 
                                                                                        'fatalities':fit['fatalities'],
                                                                                        'actual':true_obs,
                                                                                        'CRPS':crps})

            # baseline 4
            elif j == 3:
                if country == w_minimization_list[i]['minWData'][j][index].loc[0,'country_id']:
                    for s in s_prediction_list:
                        w = w_minimization_list[i]['minWData'][j][index].loc[s-3,'w']
                        fit = nBinom_quantiles(features, w, quantiles) # calculate the quantiles for the w

                        true_obs = actuals.iloc[s-3,0] # true observation of the month s
                        NB_prediction = fit['fatalities'] # value of the quantiles
                        crps = pscore(np.array(NB_prediction),true_obs).compute()[0] # compute crps

                        baseline_prediction_list[j][index]['prediction'][data_year].append({'s':s, 'w':w, 
                                                                                            'dist':fit['dist'], 'mean':fit['mean'],
                                                                                            'var':fit['var'], 
                                                                                            'quantile':string_quantile_list, 
                                                                                            'fatalities':fit['fatalities'],
                                                                                            'actual':true_obs,
                                                                                            'CRPS':crps})
                else:
                    print('Stopp')
                    break



baseline1_average_crps_list = []
baseline2_average_crps_list = []
baseline3_average_crps_list = []
baseline4_average_crps_list = []

for i in range(number_dataframes):
    year = actual_years[i]
    for index in range(number_countries):
        baseline1_average_crps_list.append(np.mean(baseline_prediction_list[0][index]['prediction'][year][0]['CRPS'])) #crps is stored as list and not individual values
        baseline2_average_crps_list.append(np.mean(baseline_prediction_list[1][index]['prediction'][year][0]['CRPS'])) #crps is stored as list and not individual values

        for s in s_prediction_list:
            baseline3_average_crps_list.append(np.mean(baseline_prediction_list[2][index]['prediction'][year][s-3]['CRPS']))
            baseline4_average_crps_list.append(np.mean(baseline_prediction_list[3][index]['prediction'][year][s-3]['CRPS']))

baseline1_average_crps = np.mean(baseline1_average_crps_list)
baseline2_average_crps = np.mean(baseline2_average_crps_list)
baseline3_average_crps = np.mean(baseline3_average_crps_list)
baseline4_average_crps = np.mean(baseline4_average_crps_list)

print('Overall CRPS')
print('baseline 1: ' + str(np.round(baseline1_average_crps, decimals = 4)))
print('baseline 2: ' + str(np.round(baseline2_average_crps, decimals = 4)))
print('baseline 3: ' + str(np.round(baseline3_average_crps, decimals = 4)))
print('baseline 4: ' + str(np.round(baseline4_average_crps, decimals = 4)))



from joblib import dump, load

# save variables in joblib file
dump([country_list, baseline_estimate_list, task2_baseline_list, w_minimization_list, baseline_prediction_list,
       baseline1_average_crps, baseline2_average_crps, baseline3_average_crps, baseline4_average_crps], 
       'task2_baseline_variables.joblib')
