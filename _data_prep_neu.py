import pandas as pd
import pickle
import numpy as np
import statsmodels.api as sm
import conversion as conv
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

'''tick1'''
# Read in .dta film data
film_data = pd.read_stata("data/films_day.dta") 
# "Drop movies which are dropped before their sixth weekend"
film_data = film_data[film_data['dropped'] != 1]

# Create opening saturday date
conditions = [
    film_data['wk1'] == 1,  
    film_data['wk2'] == 1,  
    film_data["wk3"] == 1,  
    film_data["wk4"] == 1,
    film_data["wk5"] == 1,  
    film_data["wk6"] == 1  
]
values = [
    film_data['sat_date'],     
    film_data['sat_date']-7,
    film_data['sat_date']-7*2, 
    film_data['sat_date']-7*3,
    film_data['sat_date']-7*4, 
    film_data['sat_date']-7*5]
default = pd.NaT
film_data['opening_sat_date']=np.select(conditions, values, default=default)

# Aggregation
#(take sum) tick* theaters rh1000 rl1000 vtop1000
sum_cols = [x for x in film_data.keys() if x.startswith('tick')] + ['theaters', 'rh1000', 'rl1000', 'vtop1000']
#(take mean) h* wk* dow sat_date probdropped_wk*
mean_cols = [x for x in film_data.keys() if x.startswith('h') or x.startswith('wk') or x.startswith('probdropped_wk')]+['dow','sat_date']
agg_dict = {}
for s in sum_cols:
    agg_dict[s] = 'sum'
for m in mean_cols:
    agg_dict[m] = 'mean'
film_collapsed = film_data.groupby(['opening_sat_date', 'date']).agg(agg_dict).reset_index()
# print(film_collapsed[['opening_sat_date', 'date','theaters','dow','wk1','wkintheaters','tickets_adult']].head())


# Process film data, calculate rh1000, rl1000, vtop1000 per theater
# 'Create tickets per theater measures'
for col_name in ['rh1000', 'rl1000', 'vtop1000']:
    film_collapsed[col_name+'_o'] = film_collapsed[col_name]
    film_collapsed[col_name] = film_collapsed[col_name+'_o'] / film_collapsed['theaters']
    film_collapsed.drop(columns=[col_name+'_o'], inplace=True)

with open('mod_pkl/tick_openwkend_day1.pkl','wb') as file:
    pickle.dump(film_collapsed,file)

'''tick2'''

"""
'Add holidays from opening weekend to film_collapsed'
"""

holiday_data = pd.read_stata("data/holidays.dta") 
holiday_data["dow"] = holiday_data['date'].apply(lambda x: conv.get_day_of_week(x))

# Conditions and corresponding values for opening_sat_date; Create saturday date for holiday data
conditions = [
    holiday_data['dow'] == 6,  # Saturday
    holiday_data['dow'] == 5,  # Friday
    holiday_data["dow"] == 0,  # Sunday
    holiday_data["dow"] == 1   # Monday
]
values = [
    holiday_data['date'],     # Saturday
    holiday_data['date']+1,  # Friday
    holiday_data['date']-1,  # Sunday
    holiday_data['date']-2   # Monday
]

default = pd.NaT

holiday_data['sat_date'] = np.select(conditions, values, default=default)
# print(holiday_data[['date','dow','sat_date']].head(10))

# Drop rows in column 'sat_date' with NaN values
holiday_data.dropna(subset=['sat_date'], inplace=True)
# print(holiday_data[['date','dow','sat_date']].head(10))
h_cols = [x for x in holiday_data.keys() if x.startswith("h")]

# For each opening saturday (aggregated), sum holidays that week 
grouped_sum = holiday_data[['sat_date']+h_cols].groupby('sat_date').sum().reset_index()
for h in h_cols:
    holiday_data[h] = holiday_data[h].apply(lambda x: 1 if x >0 else 0)
# print(grouped_sum.head(10))

with open('mod_pkl/tick_openwkend_day1.pkl','rb') as file:
    film_collapsed = pickle.load(file)
film_collapsed.drop(columns=[x for x in film_collapsed.keys() if x.startswith('h')],inplace=True)
# print('film_collapsed keys:',film_collapsed.keys())

# Merge: intersection
merged_data = pd.merge(film_collapsed,holiday_data,how='inner')
# print(f'{merged_data[merged_data[h_cols].isna().any(axis=1)]}')
for h in h_cols:
    merged_data[h] = merged_data[h].apply(lambda x: 1 if x >0 else 0)
# print(f'{merged_data[merged_data[h_cols].isna().any(axis=1)]}')
print('Merging complete:\n')
print(f'\t#obs in films_day: {film_collapsed.shape[0]},\n\
      #obs in holidays_p: {holiday_data.shape[0]},\n\
      #obs after merging: {merged_data.shape[0]} ')

film_collapsed = merged_data


"""
'make theater measures consistent within weekend'
"""
film_collapsed['tmp'] = film_collapsed['theaters']
film_collapsed['theaters'] = film_collapsed.groupby(['opening_sat_date','sat_date'])['tmp'].transform('max')
film_collapsed.drop(columns=['tmp'], inplace=True)


"""
"create opening weekend ticket measures": theaterso: max theaters wk1
"""
film_collapsed['tmp'] = film_collapsed['theaters'].where(film_collapsed['wk1'] == 1)
film_collapsed['theaterso'] = film_collapsed.groupby('opening_sat_date')['tmp'].transform('max')
film_collapsed.drop(columns=['tmp'], inplace=True)
# film_data = film_collapsed.sort_values(by='sat_date')
# print(film_data[['sat_date','theaters','theaterso','wk1','wk2','wk3']].head(10))

# print(film_collapsed[['o_sat_date','sat_date','wk1','wk2','wk3','wk4']].head(10))
# rows that are not o.w. also gets a theaterso value?????????
# print(film_data.shape[0])


"""
"create tickets per opening theater, per theater"
"""
film_collapsed['tickets_pot'] = film_collapsed['tickets']*10000/film_collapsed['theaterso']
film_collapsed['tickets_pt'] = film_collapsed['tickets']*10000/film_collapsed['theaters']

with open('mod_pkl/tick_openwkend_day2.pkl','wb') as file:
    pickle.dump(film_collapsed,file)


'''tick3'''

"""
merge weather_collapsed_day
"""
with open('mod_pkl/tick_openwkend_day2.pkl','rb') as file:
    film_data = pickle.load(file)
weather_data = pd.read_stata('data/weather_collapsed_day.dta')
weather_data['date'] = weather_data['date'].apply(lambda x: conv.date_to_stata(x))
weather_data['sat_date'] = weather_data['sat_date'].apply(lambda x: conv.date_to_stata(x))
weather_merged = pd.merge(film_data,weather_data,how='left')
# ***in the original paper, the merging method seems to be self defined***

# Rename {weather} to be 'own_{weather}'
w_cols = [x for x in weather_data.keys() if (not (x =="date")) and (not(x=="sat_date")) ]
for key in w_cols:
    weather_merged = weather_merged.rename(columns={key:'own_'+key})
# print(weather_merged.head(10))

with open("mod_pkl/tick_openwkend_day3.pkl",'wb') as file:
    pickle.dump(weather_merged,file)
# weather_merged.to_excel('tick3_m.xlsx',index=False)


'''tick4'''

"""
merge weather_collapsed_all
"""

with open("mod_pkl/tick_openwkend_day3.pkl",'rb') as file:
    tick_openwkend_day3 = pickle.load(file)
weather_all = pd.read_stata("data/weather_collapsed_all.dta")
# Some modification to the weather_all dataset
weather_all['sat_date'] = weather_all['sat_date'].apply(lambda x: conv.date_to_stata(x))
weather_all.rename(columns={'sat_date': 'opening_sat_date'}, inplace=True)
# print(weather_all.head(10))

# Merge weather_all with previously processed data
weather_all_m = pd.merge(tick_openwkend_day3,weather_all,how="left")

# Rename {weather_all} to be 'open_{weather_all}'
w_cols = [x for x in weather_all.keys() if (not (x =="opening_sat_date")) ]
for key in w_cols:
    weather_all_m = weather_all_m.rename(columns={key:'open_'+key})
weather_all_m['opening_sat_date'] = weather_all_m['opening_sat_date'].apply(lambda x: conv.stata_to_date(x))

# Get week and year
weather_all_m['week']=weather_all_m['opening_sat_date'].dt.isocalendar().week
weather_all_m['year']=weather_all_m['opening_sat_date'].dt.year

# Generate dummy variables for week year and day of week
weather_all_m = pd.get_dummies(weather_all_m, columns=['week'], prefix='ww', dtype=int)
weather_all_m = pd.get_dummies(weather_all_m, columns=['year'], prefix='yy', dtype=int)
weather_all_m = pd.get_dummies(weather_all_m, columns=['dow'], prefix='dow', dtype=int)
# print(weather_all_m[[x for x in weather_all_m.keys() if x.startswith('ww')]].head(10))

with open("mod_pkl/tick_openwkend_day4.pkl",'wb') as file:
    pickle.dump(weather_all_m,file)

'''prepare instruments etc'''
"""
get residuals// prepare the instruments

"""
with open("mod_pkl/tick_openwkend_day4.pkl", "rb") as file:
    data = pickle.load(file)
    
# !!!DO NOT GENERATE EXCEL FILE FOR DFs THAT ARE TOO LARGE!!! 
# (for example 1000 rows for this variable data is fine)
# data.head(1000).to_excel('data_processed_m.xlsx', index=False)

# Generate weather variables (temperature) in 10F gap
for i in range(1, 10):
    data["own_mat10_" + str(i) + "0"] = (
        data["own_mat5_" + str(i) + "0"]
        + data["own_mat5_" + str(i) + "5"]
    )
# print(data[['own_mat10_10','own_mat5_10','own_mat5_15']])

# Base case fixed effects (seasonality and holiday): control1
control1 = [
    x
    for x in data.keys()
    if x.startswith("ww")
    or x.startswith("yy")
    or x.startswith("h")
    or x.startswith("dow_")
]
# print(control1)

# Outcome variables
tickets_of_interest = [
    "tickets",
    "tickets_ratedgpg",
    "tickets_adult",
    "tickets_p33_highbudget",
    "tickets_p33_lowbudget",
    "tickets_p33_hr1000", 
    "tickets_p33_lr1000",
    "tickets_pt",
    "tickets_pot"
]


''' "get residual tickets by day for all types of tickets, wk1 only" '''
# Regenerate dow
conditions =[data['dow_5.0']==1,
             data['dow_6.0']==1,
             data['dow_0.0']==1]
values = [5,6,0]
default = pd.NaT
data['dow'] = np.select(conditions,values,default=default)

# Residualize week 1 ticket measures (without weather as control variables)
for t in tickets_of_interest:
    filtered_data = data[
        (data["wk1"] == 1) & (data[t] > 0)
    ].copy()  # Make a copy to avoid SettingWithCopyWarning
    # Save the non-numerical columns
    texts_col = filtered_data[['opening_sat_date','dow']] 
    # Extract the numerical columns of interest
    filtered_data = filtered_data[control1+[t]]
    # Prepare the data for regression, remove n.a., replace True and False with 1/0; everything to numeric
    filtered_data.fillna(0, inplace=True)
    filtered_data = filtered_data.replace({True: 1, False: 0})
    filtered_data = filtered_data.apply(pd.to_numeric, errors="coerce")
    # Regression
    y = filtered_data[t]
    X = filtered_data[control1] 
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    # Get residual
    filtered_data["tmp"] = model.resid
    # Retrieve non-numeric columns
    filtered_data['opening_sat_date'] = texts_col['opening_sat_date']
    filtered_data['dow'] = texts_col['dow']
    # Take max/sum for observations in the same group
    filtered_data['_wk1d_r'] = filtered_data.groupby(['opening_sat_date','dow'])['tmp'].transform('max') 
    filtered_data['_wk1_r'] = filtered_data.groupby('opening_sat_date')['tmp'].transform('sum')
    # Save the residuals in the original dataframe
    data.loc[filtered_data.index, t + "_wk1d_r"] = filtered_data["_wk1d_r"]
    data.loc[filtered_data.index, t + "_wk1_r"] = filtered_data["_wk1_r"]
    filtered_data.drop(columns=["tmp"], inplace=True)
    data[t + "_wk1d_r"] = data.groupby(['opening_sat_date','dow'])[t + "_wk1d_r"].transform('max')
    data[t + "_wk1_r"] = data.groupby(['opening_sat_date','dow'])[t + "_wk1_r"].transform('max')
# Check
assert(len([x for x in data.keys() if x.endswith("_wk1d_r")]) == len(tickets_of_interest))

# Residualization: analog to the code chunk above
var_list = ['rh1000', 'rl1000', 'vtop1000']
for t in var_list:
    filtered_data = data[
        (data["wk1"] == 1)
    ].copy()  # Make a copy to avoid SettingWithCopyWarning
    texts_col = filtered_data[['opening_sat_date','dow']] # TODO: , 'dow' where is dow ?????????? also fix the line below with grouping by dow
    filtered_data = filtered_data[control1+[t]]
    filtered_data.fillna(0, inplace=True)
    filtered_data = filtered_data.replace({True: 1, False: 0})
    filtered_data = filtered_data.apply(pd.to_numeric, errors="coerce")
    y = filtered_data[t]
    X = filtered_data[control1]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    filtered_data["tmp"] = model.resid
    filtered_data['opening_sat_date'] = texts_col['opening_sat_date']
    filtered_data['dow'] = texts_col['dow']
    filtered_data['_wk1d_r'] = filtered_data.groupby(['opening_sat_date','dow'])['tmp'].transform('max') #, 'dow'???????????why also group by dow? wouldn't it just be a single value for each group?
    filtered_data['_wk1_r'] = filtered_data.groupby('opening_sat_date')['tmp'].transform('sum')
    # print(filtered_data.head(30))
    data.loc[filtered_data.index, t + "_wk1d_r"] = filtered_data["_wk1d_r"]
    data.loc[filtered_data.index, t + "_wk1_r"] = filtered_data["_wk1_r"]
    filtered_data.drop(columns=["tmp"], inplace=True)
    data[t + "_wk1d_r"] = data.groupby(['opening_sat_date','dow'])[t + "_wk1d_r"].transform('max')
    data[t + "_wk1_r"] = data.groupby(['opening_sat_date','dow'])[t + "_wk1_r"].transform('max')

assert(len([x for x in data.keys() if x.endswith("_wk1d_r")]) == len(tickets_of_interest) + len(var_list))


# data.head(1000).to_excel("weather_m_tick_res.xlsx",index = False)

'''
"get residual weather"
'''
weathers = [x for x in data.keys() if x.startswith("own_")]
for w in weathers:
    # Get x and y, everything to numeric--> prepare for regression
    y = (data[w]).apply(pd.to_numeric, errors="coerce")
    X = (data[control1])
    X = X.replace({True: 1, False: 0})
    X=X.apply(pd.to_numeric, errors="coerce")
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    # print(model.summary())
    # print("Coefficients:\n", model.params)
    data['res_'+w] = model.resid
    # print(weather_all_m['res_'+w])
assert(len([x for x in data.keys() if x.startswith("res_")])==len(weathers))


'''
"create weather residuals by date"
'''

for w in weathers:
    var_name = "res_"+w
    # Create temporary columns and calculate max for dow==5, dow==6, and dow==0, then group by 'sat_date'
    data = data.replace({True: 1, False: 0})
    data['tmp_5'] = data[var_name].where(data['dow_5.0'] == 1)
    data[var_name+'_5'] = data.groupby('sat_date')['tmp_5'].transform('max')

    data['tmp_6'] = data[var_name].where(data['dow_6.0'] == 1)
    data[var_name+'_6'] = data.groupby('sat_date')['tmp_6'].transform('max')

    data['tmp_0'] = data[var_name].where(data['dow_0.0'] == 1)
    data[var_name+'_0'] = data.groupby('sat_date')['tmp_0'].transform('max')

    # print(data[[var_name,var_name+'_5','tmp_5','sat_date']].head(10))
    # Drop temporary columns
    data.drop(columns=['tmp_5', 'tmp_6', 'tmp_0'], inplace=True)



'''
"create opening weather residuals"
'''
weathers_r = [x for x in data.keys() if x.startswith('res_own')]
open_ = "open_"
for w in weathers_r:
    # Create temporary columns and calculate max for dow==5, dow==6, and dow==0, then group by 'opening_sat_date'
    data['tmp_5'] = data[w].where((data['dow_5.0'] == 1) & (data['wk1'] == 1) )
    data[open_+w+'_5'] = data.groupby('opening_sat_date')['tmp_5'].transform('max')

    data['tmp_6'] = data[w].where((data['dow_6.0'] == 1) & (data['wk1'] == 1 ))
    data[open_+w+'_6'] = data.groupby('opening_sat_date')['tmp_6'].transform('max')

    data['tmp_0'] = data[w].where((data['dow_0.0'] == 1) & (data['wk1'] == 1) )
    data[open_+w+'_0'] = data.groupby('opening_sat_date')['tmp_0'].transform('max')

# data.head(100).to_excel("weather_end.xlsx",index=False)
with open('mod_pkl/for_analyses.pkl','wb') as file:
    pickle.dump(data,file)
# print(data.keys())
print("==================complete=====================")