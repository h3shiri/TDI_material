# coding: utf-8
# ## This notebook contains calculations for the TDI challenge first part
# In[404]:

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras # for machine learning models
import scipy # linear models..etc

#Taking both time inputs as strings and returning float
import datetime
#We import the required libraries for data exploration and visualization.
import matplotlib.pyplot as plt
import os
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

print(os.listdir("."))
# In[4]:
PATH_TO_META_SAMPLES = 'Incidents_Responded_to_by_Fire_Companies.csv'
data_df = pd.read_csv(PATH_TO_META_SAMPLES)

# In[5]:

data_df.head()
# #### Exploring the different types of incidents

# In[17]:

ax = plt.figure(figsize=(30, 8))
sns.countplot(data_df.INCIDENT_TYPE_KEY)
axis_font = {'fontname':'Arial', 'size':'24'}
plt.xlabel('age',  **axis_font)
plt.ylabel('Count',  **axis_font)

# In[11]:
test = ["300 - Rescue, EMS incident, other", "735A - Unwarranted alarm/defective condition o..."]
for t in test :
    print(t.split("-")[0].strip())

# In[13]:


data_df['INCIDENT_TYPE_KEY'] = [t.split("-")[0].strip() for t in data_df['INCIDENT_TYPE_DESC']]
# In[28]:

len(data_df.INCIDENT_TYPE_KEY.unique())
# In[29]:
Code_Frequencies = data_df.INCIDENT_TYPE_KEY.value_counts()
# In[46]:
print("The leading special code is {0} with frequency {1}".format(Code_Frequencies.idxmax(), Code_Frequencies[Code_Frequencies.idxmax]))

# #### Printing results for section I
# #### Question 1 exploration
# In[50]:

#Question (i)
res = Code_Frequencies['300']/sum(Code_Frequencies.values)
format(res, '.10f')
# In[52]:

#Non cleaned result would be
temp_arr = data_df.INCIDENT_TYPE_DESC.value_counts()
res2 = temp_arr[temp_arr.idxmax]/sum(temp_arr.values)
format(res2, '.10f')
# #### Question 2 exploration
# In[65]:

#Question (ii)
data_111 = data_df.loc[data_df['INCIDENT_TYPE_KEY'] == '111']
data_651 = data_df.loc[data_df['INCIDENT_TYPE_KEY'] == '651']
# In[61]:
res = data_111['UNITS_ONSCENE'].mean()/data_651['UNITS_ONSCENE'].mean()
format(res, '.10f')

# In[64]:

data_111['UNITS_ONSCENE'].mean()

# In[66]:

data_df.BOROUGH_DESC.unique()

# #### Question 3 exploration

# In[74]:


#Question (iii)
data_Manhattan = data_df.loc[data_df['BOROUGH_DESC'] == '1 - Manhattan']
data_Staten = data_df.loc[data_df['BOROUGH_DESC'] == '3 - Staten Island']


# In[79]:


Manhattan_code_Frequencies = data_Manhattan.INCIDENT_TYPE_KEY.value_counts()
Manhattan_Mischievious_calls_rate = Manhattan_code_Frequencies['710']/sum(Manhattan_code_Frequencies.values)
Staten_code_Frequencies = data_Staten.INCIDENT_TYPE_KEY.value_counts()
Staten_Mischievious_calls_rate = Staten_code_Frequencies['710']/sum(Staten_code_Frequencies.values)
res = Staten_Mischievious_calls_rate/Manhattan_Mischievious_calls_rate
format(res, '.10f')


# #### Question 4 exploration
# In[231]:

for x in data_111.INCIDENT_DATE_TIME[0:1]:
    arr = x.split()
    str1 = x
for y in data_111.ARRIVAL_DATE_TIME[0:1]:
    arr = y.split()
    str2 = y
print(arr)
print(type(str1))
print(str1)

# In[229]:

import datetime
format = '%m/%d/%Y %I:%M:%S %p'
startDateTime = datetime.datetime.strptime(str1, format)
endDateTime = datetime.datetime.strptime(str2, format)

# In[149]:
def calcTimeDiff(arrivalTime, endTime):
    format = '%m/%d/%Y %I:%M:%S %p'
    startDateTime = datetime.datetime.strptime(arrivalTime, format)
    endDateTime = datetime.datetime.strptime(endTime, format)
    diff = endDateTime - startDateTime
    return diff.total_seconds()/60

# In[182]:
calcTimeDiff(str1, str2)

# In[183]:

c_ls = list(zip(data_111['INCIDENT_DATE_TIME'], data_111['ARRIVAL_DATE_TIME']))

# In[216]:

#We have 20 places where the code would not make sense, aka reciving a NaN value
time_diff = []
index = 0
for x in c_ls:
    if ((not isinstance(x[0], str)) or (not isinstance(x[1], str))):
        temp = 0
    else:
        temp = calcTimeDiff(x[0],x[1])
    time_diff.append(temp)
    index += 1
print(index)
print(len(data_111['INCIDENT_DATE_TIME']))
# In[223]:

# Result to question (iv)
np.percentile(time_diff, 75)


# In[217]:
data_111['TIME_DIFF_FROM_ALARM'] = time_diff

# #### Question 5 exploration

# In[240]:

def fetchHourInfo(stringInfo):
    format = '%m/%d/%Y %I:%M:%S %p'
    timeObject = datetime.datetime.strptime(stringInfo, format)
    return timeObject.hour

# In[242]:

fetchHourInfo(str2)
# In[243]:


#Here we add the hour of the day information to each incident
data_df["INCIDENT_HOUR"] = [fetchHourInfo(x) for x in data_df['INCIDENT_DATE_TIME']]

# In[244]:


data_df.head()
# In[246]:


total_incidentss_per_hour = data_df.INCIDENT_HOUR.value_counts()
# In[251]:

data_113 = data_df.loc[data_df['INCIDENT_TYPE_KEY'] == '113']
# In[254]:

cooking_fires_by_hour_count = data_113.INCIDENT_HOUR.value_counts()


# In[255]:


prop_by_hour_cooking = dict()
#Creating the relevant slices to different hours
for h in range(24):
    prop_by_hour_cooking[str(h)] = cooking_fires_by_hour_count[h]/total_incidentss_per_hour[h]

# In[266]:

res5 = max(prop_by_hour_cooking.values())
np.around(res5, 10)
# In[268]:

#Most cooking done during the day at hour 
k = list(prop_by_hour_cooking.keys())
v = list(prop_by_hour_cooking.values())
k[v.index(max(v))]
# #### Question 6 exploration

# In[269]:
PATH_TO_CENSUS_DATA = '2010+Census+Population+By+Zipcode+(ZCTA).csv'
data_US_Census = pd.read_csv(PATH_TO_CENSUS_DATA)
# In[279]:
set_of_valid_codes = data_US_Census['Zip Code ZCTA'].unique()
# In[287]:
np.shape(set_of_valid_codes)
# In[290]:
len(data_111_limited_to_census['ZIP_CODE'])
# In[316]:

type(data_US_Census[data_US_Census['Zip Code ZCTA']==1001]['2010 Census Population'].tolist()[0])
# In[318]:
for x in range(1001,1004):
    print(data_US_Census[data_US_Census['Zip Code ZCTA']==x]['2010 Census Population'].tolist()[0])

# In[289]:

data_111_limited_to_census = data_111[data_111['ZIP_CODE'].isin(set_of_valid_codes)]

# In[292]:

zip_Code_to_number_of_fires = data_111_limited_to_census.ZIP_CODE.value_counts()

# In[294]:
data_US_Census['Zip Code ZCTA'][11207.0]

# In[319]:
dict_census_data = dict() 
for zipCode in zip_Code_to_number_of_fires.keys():
    dict_census_data[zipCode] = data_US_Census[data_US_Census['Zip Code ZCTA']==zipCode]['2010 Census Population'].tolist()[0]

# In[333]:

dict_fires_per_zip = dict()
for zipCode in zip_Code_to_number_of_fires.keys():
    dict_fires_per_zip[zipCode]= zip_Code_to_number_of_fires[zipCode]
dict_fires_per_zip

# In[331]:
import scipy
def rsquared(x, y):
    """ Return R^2 where x and y are array-like."""

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    return r_value**2

# In[342]:
res6 = rsquared(list(dict_census_data.values()), list(dict_fires_per_zip.values()))
np.around(res6, 10)

# #### Question 7 exploration

# In[347]:
data_filtered_to_CO_detector = data_df[data_df['CO_DETECTOR_PRESENT_DESC'].isin(['Yes','No'])]

# In[385]:

#Building the time intervals buckets, for CO_present/Not Present
data_CO_present = data_filtered_to_CO_detector.loc[data_filtered_to_CO_detector['CO_DETECTOR_PRESENT_DESC'] == 'Yes']
data_CO_absent = data_filtered_to_CO_detector.loc[data_filtered_to_CO_detector['CO_DETECTOR_PRESENT_DESC'] == 'No']
# In[386]:

#filtering Duration of incidents and time
#Dividing to time buckets 20-30, 30-40, 40-50 ,50-60
Incidents_per_interval_CO_presnt = dict()
Incidents_per_interval_CO_absent = dict()
for x in range(65,70):
    Incidents_per_interval_CO_presnt[chr(x)] = 0
    Incidents_per_interval_CO_absent[chr(x)] = 0

# In[387]:
mis_fits = 0
for t_seconds in data_CO_present['TOTAL_INCIDENT_DURATION'] :
    if (20 <= (t_seconds/60) <= 30):
        Incidents_per_interval_CO_presnt['A'] += 1
    elif (30 <= (t_seconds/60) <= 40):
        Incidents_per_interval_CO_presnt['B'] += 1
    elif (40 <= (t_seconds/60) <= 50):
        Incidents_per_interval_CO_presnt['C'] += 1
    elif (50 <= (t_seconds/60) <= 60):
        Incidents_per_interval_CO_presnt['D'] += 1
    elif (60 <= (t_seconds/60) <= 70):
        Incidents_per_interval_CO_presnt['E'] += 1
    else:
        mis_fits += 1


# In[388]:
mis_fits_missing = 0
for t_seconds in data_CO_absent['TOTAL_INCIDENT_DURATION'] :
    if (20 <= (t_seconds/60) <= 30):
        Incidents_per_interval_CO_absent['A'] += 1
    elif (30 <= (t_seconds/60) <= 40):
        Incidents_per_interval_CO_absent['B'] += 1
    elif (40 <= (t_seconds/60) <= 50):
        Incidents_per_interval_CO_absent['C'] += 1
    elif (50 <= (t_seconds/60) <= 60):
        Incidents_per_interval_CO_absent['D'] += 1
    elif (60 <= (t_seconds/60) <= 70):
        Incidents_per_interval_CO_absent['E'] += 1
    else:
        mis_fits_missing += 1

# In[395]:
Frequency_for_intervals_present = dict()
Frequency_for_intervals_absent = dict()
target_ratio_per_bucket = dict()
total_present = len(data_CO_present.TOTAL_INCIDENT_DURATION)
total_absent = len(data_CO_absent.TOTAL_INCIDENT_DURATION)
for x in range(65,70):
    Frequncy_for_intervals_present[chr(x)] = Incidents_per_interval_CO_presnt[chr(x)]/total_present
    Frequency_for_intervals_absent[chr(x)] = Incidents_per_interval_CO_absent[chr(x)]/total_absent
    target_ratio_per_bucket[chr(x)] = Frequency_for_intervals_absent[chr(x)]/Frequncy_for_intervals_present[chr(x)]

# In[396]:
Frequncy_for_intervals_present
# In[397]:
Frequency_for_intervals_absent
# In[398]:
target_ratio_per_bucket
# In[399]:
mid_bins = []
for duration in range(25, 75, 10):
    mid_bins.append(duration)
# In[400]:
mid_bins
# In[402]:
y_vals = list(target_ratio_per_bucket.values())
# In[403]:
y_vals
# In[405]:
def predict(x, y, val):
    """ Return predition where x and y are array-like, val is querry value"""
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    return intercept + (slope* val)
# In[408]:
predict(mid_bins, y_vals, 55)
# In[409]:
res7 = predict(mid_bins, y_vals, 39)
np.round(res7, 10)