import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats



def timechange(sdf):
    sdf["time"] = pd.to_datetime(sdf["time"])
    sdf["hour"] = sdf["time"].dt.hour
    sdf["time"] = sdf["time"].dt.date

    return sdf


# create sdf with statistical values from csv 
def features(sdf):
    # create list in order to store values in for loop
    base_stats = []
    # loop through rows of sdf
    for var in sdf.variable.unique(): # var are the distinct features in the column named 'variable'
        subsetseries = sdf[sdf["variable"] == var].value # subsetseries are the numbers given in column named 'value'
        description_dict = subsetseries.describe().round(2).to_dict() # description_dict is returning a dict consisting of count, mean, std, min, 25%, 50%, 75%, and max values          
        description_dict['trimmed mean'] = stats.trim_mean(subsetseries, 0.05) 
        description_dict["variable"] = var
        base_stats.append(description_dict)
    # create new sdf
    base_stats_sdf = pd.DataFrame(base_stats)
    base_stats_sdf = base_stats_sdf.set_index("variable")

    return base_stats_sdf


# create sdf with statistical values from csv when appCat and circumplex values are summed
def features_aggr(sdf, aggr_features=False):


    # sum values for appCat and circumplex variables
    if aggr_features:
        sdf['variable'] = sdf['variable'].apply(lambda x: 'appCat.values' if 'appCat' in x else x)
        sdf['variable'] = sdf['variable'].apply(lambda x: 'circumplex.values' if 'circumplex' in x else x)

    # create list in order to store values in for loop
    statistics = []

    # loop through rows of sdf
    for var in sdf.variable.unique(): # var are the distinct features in the column named 'variable'
        col_val = sdf[sdf["variable"] == var].value # subsetseries are the numbers given in column named 'value'
        description_dict = col_val.describe().round(2).to_dict() # description_dict is returning a dict consisting of count, mean, std, min, 25%, 50%, 75%, and max values          
        description_dict['trimmed mean'] = stats.trim_mean(col_val, 0.05) 
        description_dict["variable"] = var
        statistics.append(description_dict)

    # create a new sdf
    statistics_sdf = pd.DataFrame(statistics)
    statistics_sdf = statistics_sdf.set_index("variable")

    return statistics_sdf


def counting_mood_days(sdf):
    # store values in here
    mood_days = []
    moodless_days = []

    # separate users
    for users in sdf.id.unique():
        # loop through individual users
        # see: https://datatofish.com/select-rows-pandas-dataframe/
        user_rows = sdf.loc[sdf.id == users]


        days_with_mood = 0
        days_without_mood = 0
        
        # iterate through rows for one user only
        for i, day in user_rows.iterrows():
            if day['variable'] == 'mood':
                days_with_mood += 1
            else:
                days_without_mood += 1
        mood_days.append({users: days_with_mood})
        moodless_days.append({users: days_without_mood})
    return(mood_days, moodless_days)


feat_dict = {
        'mood': float(0),
        'circumplex.arousal': float(0),
        'circumplex.valence': float(0),
        'activity': float(0),
        'screen': float(0),
        'sms': float(0),
        'appCat.builtin': float(0),
        'appCat.communication': float(0),
        'appCat.entertainment': float(0),
        'appCat.finance': float(0),
        'appCat.game': float(0),
        'appCat.office': float(0),
        'appCat.other': float(0),
        'appCat.social': float(0),
        'appCat.travel': float(0),
        'appCat.unknown': float(0),
        'appCat.utilities': float(0),
        'appCat.weather': float(0),
    }



# calculate average value per variable per day and keep only one per variable per day left
new_sdf = sdf.groupby(['id', 'time', 'variable'])['value'].mean().to_frame().reset_index()

# transform sdf to a pivot table --> this lines up the distinct variables into columns instead of rows
# based on: https://pandas.pydata.org/docs/user_guide/reshaping.html
sdf_pivot = new_sdf.pivot(index=['id', 'time'], columns=['variable'], values=['value'])
sdf_pivot.columns = sdf_pivot.columns.droplevel(0)
# tranform pivot table back into a dataframe to make it easier to work with
organised_sdf = sdf_pivot.reset_index().rename_axis(None, axis=1)
display(organised_sdf)