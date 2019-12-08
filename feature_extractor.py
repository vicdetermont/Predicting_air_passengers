# Selected feature extractor
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
import math 
import numpy as np
from geopy.distance import geodesic
from sklearn.preprocessing import RobustScaler


def compute_distance(X_encoded):
    return X_encoded.apply(
        lambda x: geodesic(
            (x["d_latitude_deg"],x["d_longitude_deg"]),
            (x["a_latitude_deg"],x["a_longitude_deg"])
        ).km,
        axis=1,
    )

from sklearn.preprocessing import RobustScaler

cols_to_norm = ["WeeksToDeparture", "std_wtd", "d_Max TemperatureC",
                    "d_MeanDew PointC",
                    "d_Max Humidity",
                    "d_Max Sea Level PressurehPa",
                    "d_Max VisibilityKm",
                    "d_Mean VisibilityKm",
                    "d_Min VisibilitykM",
                    "d_Max Wind SpeedKm/h",
                    "d_Mean Wind SpeedKm/h",
                    "d_Precipitationmm",
                    "d_CloudCover",
                    "d_WindDirDegrees",
                    "d_latitude_deg",
                    "d_longitude_deg",
                    "d_elevation_ft",
                    "d_2018",
                    "d_2017",
                    "d_2016",
                    "d_2015",
                    "d_2017GDPPerCapita",
                    "d_OtherAirports",
                    "d_arrival_avg_WeeksToDeparture",
                    "d_arrival_avg_stdwtd",
                    "d_arrival_avg_output",
                    "d_departure_avg_WeeksToDeparture",
                    "d_departure_avg_stdwtd",
                    "d_departure_avg_output",
                    "d_day",
                    "d_PercentageOnTime",
                    "a_Max TemperatureC",
                    "a_MeanDew PointC",
                    "a_Max Humidity",
                    "a_Max Sea Level PressurehPa",
                    "a_Max VisibilityKm",
                    "a_Mean VisibilityKm",
                    "a_Min VisibilitykM",
                    "a_Max Wind SpeedKm/h",
                    "a_Mean Wind SpeedKm/h",
                    "a_Precipitationmm",
                    "a_CloudCover",
                    "a_WindDirDegrees",
                    "a_latitude_deg",
                    "a_longitude_deg",
                    "a_elevation_ft",
                    "a_2018",
                    "a_2017",
                    "a_2016",
                    "a_2015",
                    "a_2017GDPPerCapita",
                    "a_OtherAirports",
                    "a_arrival_avg_WeeksToDeparture",
                    "a_arrival_avg_stdwtd",
                    "a_arrival_avg_output",
                    "a_departure_avg_WeeksToDeparture",
                    "a_departure_avg_stdwtd",
                    "a_departure_avg_output",
                    "a_day",
                    "a_PercentageOnTime",
                    "d_ManyFlights",
                    "a_ManyFlights",
                    "departure_arrival_interaction",
                    "Distance",
                    "year",
                    "month",
                    "day",
                    "weekday",
                    "week",
                    "n_days"]
 


class FeatureExtractor(object):
    def __init__(self):
        pass

    def fit(self, X_df, y_array):
        pass

    def transform(self, X_df):
        X_encoded = X_df
        path = os.path.dirname(__file__)
        
        ## External data processing
        external_data = pd.read_csv(os.path.join(path,'external_data.csv'))
        external_data.loc[:,"Date"] = pd.to_datetime(external_data.loc[:,"Date"])
        
        # Building column names for conditions at departure and arrival 
        col_dep = ['d_' + name for name in list(external_data.columns)]
        col_arr = [w.replace('d_', 'a_') for w in col_dep]
        
        # Fitting the names of the first 2 columns to match our original dataframe 
        col_dep = [w.replace('d_AirPort', 'Departure') for w in col_dep]
        col_dep = [w.replace('d_Date', 'DateOfDeparture') for w in col_dep]
        col_arr = [w.replace('a_AirPort', 'Arrival') for w in col_arr]
        col_arr = [w.replace('a_Date', 'DateOfDeparture') for w in col_arr]
        
        # Building 2 dataframes from data_add to get the information for the departure and arrival airports of each flight
        # Departure airport 
        external_dataDeparture = external_data.copy()
        external_dataDeparture.columns = col_dep
        # Arrival airport
        external_dataArrival = external_data.copy()
        external_dataArrival.columns = col_arr
        
        # Merging them with X_encoded 
        X_encoded = X_df.copy()
        X_encoded.loc[:,'DateOfDeparture'] = pd.to_datetime(X_encoded.loc[:,'DateOfDeparture'])
        X_encoded = pd.merge(X_encoded, external_dataDeparture, how='left',left_on=['DateOfDeparture', 'Departure'],
                             right_on=['DateOfDeparture', 'Departure'],sort=False)
        X_encoded = pd.merge(X_encoded, external_dataArrival, how='left',left_on=['DateOfDeparture', 'Arrival'],
                             right_on=['DateOfDeparture', 'Arrival'],sort=False) 
        
        ### Feature engineering
        ## Creating columns to distinguish between the two main airports for flights and the rest
        X_encoded['d_ManyFlights'] = 0  
        X_encoded['a_ManyFlights'] = 0
        X_encoded.loc[X_encoded.loc[:,'Departure'] == 'ORD', "d_ManyFlights"] = 1
        X_encoded.loc[X_encoded.loc[:,'Arrival'] == 'ORD', "a_ManyFlights"] = 1
        X_encoded.loc[X_encoded.loc[:,'Departure'] == 'ATL', "d_ManyFlights"] = 1
        X_encoded.loc[X_encoded.loc[:,'Arrival'] == 'ATL', "a_ManyFlights"] = 1

        # Getting the interaction of departure and arrival on the output
        # We inputted the average output by departure/arrival airport in external data
        X_encoded["airportTraffic_interaction"] = X_encoded.loc[:,"d_departure_avg_output"]*X_encoded.loc[:,"a_arrival_avg_output"]
        X_encoded["DistanceToHoliday_interaction"] = X_encoded.loc[:,"d_DistanceToClosestHoliday"]*X_encoded.loc[:,"a_DistanceToClosestHoliday"]
        X_encoded["GDPPerCapita_interaction"] = X_encoded.loc[:,"d_2017GDPPerCapita"]*X_encoded.loc[:,"a_2017GDPPerCapita"]
        X_encoded["Region_interaction"] = X_encoded.loc[:,"d_M"]*X_encoded.loc[:,"a_M"] + X_encoded.loc[:,"d_S"]*X_encoded.loc[:,"a_S"] + X_encoded.loc[:,"d_N"]*X_encoded.loc[:,"a_N"] + X_encoded.loc[:,"d_W"]*X_encoded.loc[:,"a_W"]
        #X_encoded["MaxTemperature_interaction"] = X_encoded.loc[:,"d_Max TemperatureC"]*X_encoded.loc[:,"a_Max TemperatureC"]

        # Distance
        X_encoded["Distance"] = compute_distance(X_encoded)

        
        ## Categorical encoding of departure and arrival airports
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded.loc[:,'Departure'], prefix='d'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded.loc[:,'Arrival'], prefix='a'))
                                   
        ## Categorical encoding of the dates 
        X_encoded['year'] = X_encoded.loc[:,'DateOfDeparture'].dt.year
        X_encoded['month'] = X_encoded.loc[:,'DateOfDeparture'].dt.month
        X_encoded['day'] = X_encoded.loc[:,'DateOfDeparture'].dt.day
        X_encoded['weekday'] = X_encoded.loc[:,'DateOfDeparture'].dt.weekday
        X_encoded['week'] = X_encoded.loc[:,'DateOfDeparture'].dt.week
        X_encoded['n_days'] = X_encoded.loc[:,'DateOfDeparture'].apply(lambda date: 
                                                                         (date - pd.to_datetime("1970-01-01")).days)
        
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['year'], prefix='y'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['month'], prefix='m'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['day'], prefix='d'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['weekday'], prefix='wd'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['week'], prefix='w'))
    
        # Finally getting rid of departure, arrival, and date columns now that we do not need them to merge
        X_encoded = X_encoded.drop('Departure', axis=1)
        X_encoded = X_encoded.drop('Arrival', axis=1)
        X_encoded = X_encoded.drop('DateOfDeparture',axis = 1)

        # Scaling the data for selected columns
        #scaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0), copy=True)
        #X_encoded[cols_to_norm] = scaler.fit_transform(X_encoded[cols_to_norm])

    
        return X_encoded