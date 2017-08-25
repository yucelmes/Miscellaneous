
# coding: utf-8

# In[ ]:

import json
import pandas as pd
import urllib.parse
import urllib.request
from bson import ObjectId
from pymongo import MongoClient
from geopy.distance import VincentyDistance
from joblib import Parallel, delayed
#import multiprocessing
#num_cores = multiprocessing.cpu_count()



client = MongoClient('mongodb://ec2-34-250-110-180.eu-west-1.compute.amazonaws.com:27017')
db = client['C8Geo']
collection = db['locations']

url = 'https://api.opencagedata.com/geocode/v1/json?q='
api_key = '&pretty=1&key=c2cea158878d4b51a79fd9aed4495e72'


ObjectIdList = json.load(open(r'./Desktop/ObjectIdList.json'))
ObjectIdList = [ObjectId(Object) for Object in ObjectIdList]
Documents = collection.find({'opencage' : {'$exists' : False}, '_id' : {'$in' : ObjectIdList[:100000]}})



def UpdateDocuments(doc):

    LatLong = str(doc['_source']['coords']['latitude']) + ',' + str(doc['_source']['coords']['longitude'] )
    Response = urllib.request.urlopen(url + LatLong + api_key)
    Response = Response.read()
    Response = Response.decode('utf-8')   
    Response = json.loads(Response)

    OpenCage = {
        'type' : Response['results'][0]['components'].get('_type', '').upper(),
        'city' : Response['results'][0]['components'].get('city', ''),
        'country' : Response['results'][0]['components'].get('country', ''),
        'countrycode' : Response['results'][0]['components'].get('country_code', '').upper(),
        'county' : Response['results'][0]['components'].get('county', ''),
        'formattedaddress' : Response['results'][0].get('formatted', ''),
        'housenumber' : Response['results'][0]['components'].get('housenumber', ''),
        'postcode' : Response['results'][0]['components'].get('postcode', ''),
        'state' : Response['results'][0]['components'].get('state', ''),
        'statedistrict' : Response['results'][0]['components'].get('state_district', ''),
        'street' : Response['results'][0]['components'].get('road', ''),
        'suburb' : Response['results'][0]['components'].get('suburb', '')
    }

    doc['opencage'] = OpenCage
    collection.save(doc)
    
    return str(doc['_id'])
    

    

if __name__ == '__main__':
    
    AnalysedIds = Parallel(n_jobs=4)(delayed(UpdateDocuments)(doc) for doc in Documents)
    
    ObjectIdList = [str(Object) for Object in ObjectIdList]
    ObjectIdList = list(set(ObjectIdList).difference(set(AnalysedIds)))
    
    with open(r'./Desktop/ObjectIdList.json', 'w') as theFile:
        json.dump(ObjectIdList, theFile)

