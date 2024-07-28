from datetime import datetime, timedelta
import pandas as pd

def date_to_stata(normal_date):
    date_only = pd.to_datetime(normal_date).date()

    stata_epoch = pd.Timestamp('1960-01-01').date()
    return (date_only - stata_epoch).days

def stata_to_date(stata_date):
    # Stata's epoch starts from January 1, 1960
    stata_epoch = datetime(1960, 1, 1)
    
    delta_days = timedelta(days=int(stata_date))
    
    normal_date = stata_epoch + delta_days
    
    return normal_date

def get_day_of_week(stata_date):
    normal_date = stata_to_date(stata_date)
    
    # default is monday=0... sunday=6, however in dataset sunday = 0
    day_of_week = (normal_date.weekday() + 1) % 7 

    return day_of_week

def txt_to_lst(text:str)->list:
    text_list = text.split()
    return [x for x in text_list if len(x)>0]

# print(txt_to_lst("own_mat10_10 own_mat10_20 own_mat10_30 own_mat10_40 own_mat10_50 own_mat10_60 own_mat10_70 own_mat10_80 own_mat10_90 own_snow own_rain own_prec_0 own_prec_1 own_prec_2 own_prec_3 own_prec_4 own_prec_5"))
# print(stata_to_date(8036))
# print(date_to_stata(stata_to_date(8036)))
