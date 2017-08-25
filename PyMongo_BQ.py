
# coding: utf-8

# In[15]:

import pandas as pd
from pymongo import MongoClient
from geopy.distance import VincentyDistance
import calendar
import datetime


client = MongoClient('mongodb://ec2-34-248-97-137.eu-west-1.compute.amazonaws.com:27017')
db = client['C8Geo']
collection = db['locations']


BQ_Locs = pd.read_excel('/opt/ds/2017-07-27 - 2017-07-30 B&Q LU.xlsx')
BQ_Locs['EastNorth'] = tuple(zip(BQ_Locs['Easting'], BQ_Locs['Northing']))
EastNorth = BQ_Locs['EastNorth'].unique()


LatLongFromEastNorth = [(51.532794634, -0.105992309244), (51.5132086993, -0.0898994175462),
          (51.514518168, -0.149137568402), (51.5051934677, -0.020852146258),
          (51.5082376844, -0.125124883546), (51.5130542929, -0.124350645866),
          (51.5073784347, -0.122566317803), (51.5276392675, -0.132185018765),
          (51.5067284159, -0.142738206891), (51.50102882, -0.19292291863),
          (51.517349387, -0.120210524011), (51.5302039659, -0.123847704098),
          (51.5018707289, -0.160888606026), (51.5112571, -0.128373196017),
          (51.505987632, -0.0882416816698), (51.5119970092, -0.157497612348),
          (51.5090632286, -0.196697653109), (51.5153702172, -0.142170522541),
          (51.5166808098, -0.176982449779), (51.5102328295, -0.135202802813),
          (51.4924875929, -0.15662770173), (51.4940737772, -0.174095742942),
          (51.5037946527, -0.113304481164), (51.4645902548, -0.0128052028752),
          (51.4815075581, -0.0108024702858)]


GeoDict = dict(zip(EastNorth, LatLongFromEastNorth))
London_Lat_Bounds = [51.0, 52.0]
London_Long_Bounds = [-1.0, 1.0]
Days = list(calendar.day_abbr)


InLocUsers_VarR = { E_N : {} for E_N in EastNorth}
#Radius = [100, 200, 300, 400, 500]
Radius = [100, 300, 500]




def BQLocationAnalysis():
    

    for doc in collection.find({'created_at': {'$exists' : True}}, projection={'_id' : False, 'created_at' : True, 'userId' : True, '_source.coords.longitude' : True, '_source.coords.latitude' : True}):

        Date = str(doc['created_at'])[:10]

        if Date > '2017-04-06':

            YearMonthDay = [int(i) for i in Date.split('-')]
            WeekDay = Days[datetime.date(YearMonthDay[0], YearMonthDay[1], YearMonthDay[2]).weekday()]

            if WeekDay in Days[-4:]:

                Coords = doc['_source']['coords']['latitude'], doc['_source']['coords']['longitude']

                if (London_Lat_Bounds[0] <= Coords[0] <= London_Lat_Bounds[1]) & (London_Long_Bounds[0] <= Coords[1] <= London_Long_Bounds[1]):

                    for TargetEastNoth, TargetLatLong in GeoDict.items():

                        for R in Radius:

                            if VincentyDistance(TargetLatLong, Coords).meters <= R:

                                NewKey = (WeekDay, R)

                                if NewKey in InLocUsers_VarR[TargetEastNoth].keys():
                                    InLocUsers_VarR[TargetEastNoth][NewKey].add(doc['userId'])
                                else:
                                    InLocUsers_VarR[TargetEastNoth][NewKey] = {doc['userId']}



    Keys_100 = [DayDist for DayDist in InLocUsers_VarR[list(InLocUsers_VarR.keys())[0]].keys() if 100 in DayDist]
    Keys_300 = [DayDist for DayDist in InLocUsers_VarR[list(InLocUsers_VarR.keys())[0]].keys() if 300 in DayDist]
    Keys_500 = [DayDist for DayDist in InLocUsers_VarR[list(InLocUsers_VarR.keys())[0]].keys() if 500 in DayDist]



    NameTots_100 = {}
    NameTots_300 = {}
    NameTots_500 = {}

    
    for LocKey in list(InLocUsers_VarR.keys()):

        DailyData = InLocUsers_VarR[LocKey]
        Users = [set(), set(), set()]

        for DayKey, DayData in DailyData.items():

            if DayKey in Keys_100:
                Users[0] = Users[0].union(DayData)
            elif DayKey in Keys_300:
                Users[1] = Users[1].union(DayData)
            elif DayKey in Keys_500:
                Users[2] = Users[2].union(DayData)

        Name = BQ_Locs[BQ_Locs['EastNorth'] == LocKey]['Station_Name'][:1].get_values()[0]
        
        NameTots_100[(Name, LocKey)] = len(Users[0])
        NameTots_300[(Name, LocKey)] = len(Users[1])
        NameTots_500[(Name, LocKey)] = len(Users[2])
        
        
    print(NameTots_100)
    print(NameTots_300)
    print(NameTots_500)
    
    pd.DataFrame(data=list(NameTots_100.values()), index=list(NameTots_100.keys()), columns=['User Number']).to_csv('/opt/ds/BQ_100m.csv')
    pd.DataFrame(data=list(NameTots_300.values()), index=list(NameTots_300.keys()), columns=['User Number']).to_csv('/opt/ds/BQ_300m.csv')
    pd.DataFrame(data=list(NameTots_500.values()), index=list(NameTots_500.keys()), columns=['User Number']).to_csv('/opt/ds/BQ_500m.csv')
    
    
    return None




if __name__ == '__main__':
    
    BQLocationAnalysis()

