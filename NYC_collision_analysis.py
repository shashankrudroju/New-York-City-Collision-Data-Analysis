'''This program applies various data mining techniques on NewYork city collisons data and infers some information.
Author : Shashank Rudroju
Email : sr1632@rit.edu'''

from math import isnan
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import collections

def main():
    df = pd.read_csv('NYPD_Motor_Vehicle_Collisions.csv', low_memory=False)  # read the input csv file
    print(df.shape)
    #print(type(df['LOCATION'].values))
    df = df[pd.notnull(df['DATE'])] # Removing records that doesnt have date field
    df = df[pd.notnull(df['LATITUDE'])] #Removing records that doesnt have date field
	#Merging some columns into one
    df['PERSON_CASUALITIES'] = df['NUMBER OF PERSONS INJURED'] + df['NUMBER OF PERSONS KILLED']
    df['PEDESTRIAN_CASUALITIES'] = df['NUMBER OF PEDESTRIANS INJURED'] + df['NUMBER OF PEDESTRIANS KILLED']
    df['CYCLIST_CASUALITIES'] = df['NUMBER OF CYCLIST INJURED'] + df['NUMBER OF CYCLIST KILLED']
    df['MOTORIST_CASUALITIES'] = df['NUMBER OF MOTORIST INJURED'] + df['NUMBER OF MOTORIST KILLED']
    df['TOTAL CASUALITIES'] = df['PERSON_CASUALITIES']+df['PEDESTRIAN_CASUALITIES']+df['CYCLIST_CASUALITIES']+df['MOTORIST_CASUALITIES']
	# Dropping irrelevant columns
    df = df.drop(['NUMBER OF PERSONS INJURED', 'NUMBER OF PERSONS KILLED','NUMBER OF PEDESTRIANS INJURED','NUMBER OF PEDESTRIANS KILLED',
                     'NUMBER OF CYCLIST INJURED','NUMBER OF CYCLIST KILLED','NUMBER OF MOTORIST INJURED','NUMBER OF MOTORIST KILLED',
                     'CONTRIBUTING FACTOR VEHICLE 2','CONTRIBUTING FACTOR VEHICLE 3','CONTRIBUTING FACTOR VEHICLE 4','CONTRIBUTING FACTOR VEHICLE 5',
                     'VEHICLE TYPE CODE 2','VEHICLE TYPE CODE 3','VEHICLE TYPE CODE 4','VEHICLE TYPE CODE 5'], axis = 1)
    print('After cleaning')
    print(df.shape)
    print(df.columns)
	
	# Visualizing location data
    x = df['LATITUDE'].values
    y = df['LONGITUDE'].values
    data = np.array((x, y))
    transformed_data = data.T
	# Applying KMeans algorithm on the data to get clusters
    kmeans = KMeans(n_clusters=5, random_state=0).fit(transformed_data)
    print(kmeans.cluster_centers_)
    cluster_map = pd.DataFrame()
    cluster_map['data_index'] = df.index.values
    cluster_map['cluster'] = kmeans.labels_
    i = 0
    for center in kmeans.cluster_centers_:
        print('Center : [', round(float(center[0]), 2), round(float(center[1]), 2), ']', 'Size of Cluster :',
              len(cluster_map[cluster_map.cluster == i]))
        i += 1

    plt.xlabel('LATITUDE')
    plt.ylabel('LONGITUDE')
    plt.title('ACCIDENTS BY LOCATION')
    plt.scatter(x, y)
    plt.show()

    # BOROUGHS
    plt.xlabel('Boroughs')
    plt.ylabel('Number of Accidents')
    plt.title('Accidents by Boroughs')
    boroughs_map = dict(df['BOROUGH'].value_counts())
    print(boroughs_map)
    plt.bar(list(boroughs_map.keys()), boroughs_map.values(), color='g')
    plt.show()

    df['Year'] = df['DATE'].str[-4:]
    df['Month'] = df['DATE'].str[0:2]
    df['Hour'] = df['TIME'].str[0:-3]

    # MONTHLY
    print('Month Counts \n',df['Month'].value_counts())
    monthly_accidents = dict(df['Month'].value_counts())
    plt.xlabel('Months')
    plt.ylabel('Number of Accidents')
    plt.title('Accidents by Months')
    print(monthly_accidents)
    plt.bar(list(monthly_accidents.keys()), monthly_accidents.values(), color='b')
    plt.show()


    # HOURLY
    print('Hour Counts \n',df['Hour'].value_counts())
    hourly_accidents = dict(df['Hour'].value_counts())
    hourly_accidents = {int(k): int(v) for k, v in hourly_accidents.items()}
    plt.xlabel('Hours')
    plt.ylabel('Number of Accidents')
    plt.title('Accidents by Hours')
    x = sorted(hourly_accidents.keys())
    print(x)
    y = []
    for i in x:
        y.append(hourly_accidents[i])
    print(y)
    plt.plot(x,y)
    plt.show()


    #Yearly
    yearly_accidents = dict(df['Year'].value_counts())
    print('Year Counts \n', df['Year'].value_counts())
    plt.xlabel('Years')
    plt.ylabel('Number of Accidents')
    plt.title('Accidents by Years')
    plt.bar(list(yearly_accidents.keys()), yearly_accidents.values(), color='r')
    plt.show()

    #CYCLIST ACCIDENTS IN JUNE, JULY and AUGUST
    june_cyclists = df.loc[df['Month'] == '06']['CYCLIST_CASUALITIES'].sum()
    print('Cycle Accidents in June_:',june_cyclists)
    july_cyclists = df.loc[df['Month'] == '07']['CYCLIST_CASUALITIES'].sum()
    print('Cycle Accidents in July :',july_cyclists)
    august_cyclists = df.loc[df['Month'] == '08']['CYCLIST_CASUALITIES'].sum()
    print('Cycle Accidents in August :',august_cyclists)
    month = ['June','July','August']
    cycle_accidents = [june_cyclists,july_cyclists,august_cyclists]
    plt.xlabel('Months')
    plt.ylabel('Number of Cycle Accidents')
    plt.title('Cycle Accidents in June July and AUgust')
    plt.bar(month, cycle_accidents, color='g')
    plt.show()

    deaths = {}
    manhattan_total = df.loc[df['BOROUGH'] == 'MANHATTAN']['TOTAL CASUALITIES'].sum()
    deaths['MANHATTAN'] = manhattan_total
    print('Deaths in Manhattan',manhattan_total)
    brooklyn_total = df.loc[df['BOROUGH'] == 'BROOKLYN']['TOTAL CASUALITIES'].sum()
    deaths['BROOKLYN']=brooklyn_total
    print('Deaths in brooklyn ',brooklyn_total)
    queens_total = df.loc[df['BOROUGH'] == 'QUEENS']['TOTAL CASUALITIES'].sum()
    deaths['QUEENS'] = queens_total
    print('Deaths in Queens ',queens_total)
    bronx_total = df.loc[df['BOROUGH'] == 'BRONX']['TOTAL CASUALITIES'].sum()
    deaths['BRONX'] = bronx_total
    print('Deaths in Bronx ',bronx_total)
    staten_total = df.loc[df['BOROUGH'] == 'STATEN ISLAND']['TOTAL CASUALITIES'].sum()
    deaths['STATEN_ISLAND'] = staten_total
    print('Deaths in Staten Island ',staten_total)
    plt.xlabel('Boroughs')
    plt.ylabel('Total number of deaths/injuries')
    plt.title('Number of deaths/injuries by Borough')
    print(deaths)
    plt.bar(list(deaths.keys()), deaths.values(), color='g')
    plt.show()

    #TYPE OF ACCIDENTS
    contributing_factors = dict(df['CONTRIBUTING FACTOR VEHICLE 1'].value_counts())
    contributing_factors = removekey(contributing_factors,'Unspecified')
    print('contributing_factors \n', df['CONTRIBUTING FACTOR VEHICLE 1'].value_counts())
    plt.xticks(rotation = 'vertical')
    plt.title('Accidents by Contributing factor')
    print(contributing_factors)
    contributing_factors_values = list(contributing_factors.values())[:-15]
    contributing_factors_keys = list(contributing_factors.keys())[:-15]
    contributing_factors2 = sum(list(contributing_factors.values())[-15:])
    new = {}
    i=0
    for key in contributing_factors_keys:
        new[key] = contributing_factors_values[i]
        i+=1
    new['Others'] = int(contributing_factors2)
    plt.pie(new.values(),labels = list(new.keys()))
    plt.show()


def removekey(d, key):
    r = dict(d)
    del r[key]
    return r

if __name__ == '__main__':
    main()
