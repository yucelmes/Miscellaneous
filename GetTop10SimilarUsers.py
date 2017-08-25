
# coding: utf-8

# In[ ]:

import psycopg2
import pandas as pd
import numpy as np
import json
import sys



def WeightedMean(Data, Weights):    
    return (Data * Weights).sum() / Weights.sum()


def WeightedCovariance(Data_1, Data_2, Weights):
    return (Weights * (Data_1 - WeightedMean(Data_1, Weights)) * (Data_2 - WeightedMean(Data_2, Weights))).sum() / Weights.sum()


def WeightedCorrelation(Data_1, Data_2, Weights):    
    return WeightedCovariance(Data_1, Data_2, Weights) / np.sqrt(WeightedCovariance(Data_1, Data_1, Weights) * WeightedCovariance(Data_2, Data_2, Weights))



conn_string = "host='c-sec-prod.cilede3fhork.eu-west-1.rds.amazonaws.com' dbname='c_db' user='mesut' password='TeW15WQBfA7i'"
conn = psycopg2.connect(conn_string)
conn.autocommit = True
cursor = conn.cursor()
        

AudienceSimilarityDictionary = json.load(open(r'.\Documents\GitHub\cayenne\AudienceSimilarityDictionary.json'))
AudienceThresholds = json.load(open(r'.\Documents\GitHub\cayenne\AudienceThresholds.json'))


cursor.execute("select user_id from integrations where type = 'spotify' and is_deleted = 'False'")
SpotifyUserIDs = cursor.fetchall()
SpotifyUserIDs = tuple(str(User[0]) for User in SpotifyUserIDs)

cursor.execute('select * from user_audiences_mean where user_id in %s', [SpotifyUserIDs])
SpotifyUsers = pd.DataFrame(cursor.fetchall())
SpotifyUsers.columns = [ColNames[0] for ColNames in cursor.description]


AudIDs = pd.DataFrame(SpotifyUsers.groupby(['audience_id'])['observation_number'].mean() >= 2)
AudIDs = AudIDs[AudIDs['observation_number'] == True].index.get_values()
AudIDsSTR = tuple(str(AudID) for AudID in AudIDs)

SpotifyUsers = SpotifyUsers[SpotifyUsers['audience_id'].isin(AudIDs)]
SpotifyUserIDs = tuple(str(User) for User in SpotifyUsers['user_id'].unique())


cursor.execute('select * from audience_question_list2')
AudienceQuestions = cursor.fetchall()


ID2Name = {}
for AudienceID, AudienceName, _, _, _, _ in AudienceQuestions:
    if not AudienceID in ID2Name.keys():
        ID2Name[AudienceID] = AudienceName
Name2ID = dict(zip(ID2Name.values(), ID2Name.keys()))


TotalUserNumber = len(SpotifyUserIDs)
TenPercent = round(TotalUserNumber/10)




def GetSimilarUsers(SourceUserID = SpotifyUserIDs[0]):

    SourceUserID = int(SourceUserID)
    N = 5
    SimilarUsers = pd.DataFrame(columns=['UserID', 'SimilarityIndex'], index=[i for i in range(N)])
    SimilarUsers[:] = 0

    UserAudienceMatrix = pd.DataFrame(columns=['SourceUserMean', 'SourceObsNumber', 'TargetUserMean', 'TargetObsNumber'], index=AudIDs)
    UserAudienceMatrix[:] = 'M'

    cursor.execute('select * from user_audiences_mean where user_id in %s and audience_id in %s', (SpotifyUserIDs, AudIDsSTR))
    UserAudienceData = pd.DataFrame(cursor.fetchall())
    UserAudienceData.columns = [ColNames[0] for ColNames in cursor.description]

    SourceUserAudValues = UserAudienceData[UserAudienceData['user_id'] == SourceUserID][['audience_id', 'observation_number', 'mean']].copy()
    
    
    if len(SourceUserAudValues) >= 200:
        
        UserAudienceData = UserAudienceData[UserAudienceData['user_id'] != SourceUserID]
        MissingAudIDs = []


        for Aud_ID in UserAudienceMatrix.index:
            try:
                UserAudienceMatrix.loc[Aud_ID, ['SourceUserMean', 'SourceObsNumber']] = SourceUserAudValues[SourceUserAudValues['audience_id'] == Aud_ID][['mean', 'observation_number']].get_values()
            except:
                MissingAudIDs.append(Aud_ID)


        for Missing_AudID in MissingAudIDs:

            try:
                SimilarAudiences = AudienceSimilarityDictionary[ID2Name[Missing_AudID]]

                for SimilarAud in SimilarAudiences:            
                    Sim_AudID = Name2ID[SimilarAud[0]]
                    if (Sim_AudID in UserAudienceMatrix.index):
                        if (UserAudienceMatrix['SourceUserMean'].loc[Sim_AudID] != 'M'):
                            if SimilarAud[2] == '+':
                                UserAudienceMatrix['SourceUserMean'].loc[Missing_AudID] = UserAudienceMatrix['SourceUserMean'].loc[Sim_AudID]
                                UserAudienceMatrix['SourceObsNumber'].loc[Missing_AudID] = 1
                                break
                            else:
                                UserAudienceMatrix['SourceUserMean'].loc[Missing_AudID] = 1 - UserAudienceMatrix['SourceUserMean'].loc[Sim_AudID]
                                UserAudienceMatrix['SourceObsNumber'].loc[Missing_AudID] = 1
                                break
            except:
                continue




        for UserNumber, Target_UserID in enumerate(SpotifyUserIDs):

            Target_UserID = int(Target_UserID)

            if Target_UserID != SourceUserID:


                UserAudienceMatrix['TargetUserMean'] = 'M'
                UserAudienceMatrix['TargetObsNumber'] = 'M'
                TargetUserAudValues = UserAudienceData[UserAudienceData['user_id'] == Target_UserID][['audience_id', 'observation_number', 'mean']].copy()
                UserAudienceData = UserAudienceData[UserAudienceData['user_id'] != Target_UserID]


                if len(TargetUserAudValues) >= 200:

                    MissingAudIDs = []

                    for Aud_ID in UserAudienceMatrix.index:

                        try:
                            UserAudienceMatrix.loc[Aud_ID, ['TargetUserMean', 'TargetObsNumber']] = TargetUserAudValues[TargetUserAudValues['audience_id'] == Aud_ID][['mean', 'observation_number']].get_values()
                        except:
                            MissingAudIDs.append(Aud_ID)


                    for Missing_AudID in MissingAudIDs:

                        try:
                            SimilarAudiences = AudienceSimilarityDictionary[ID2Name[Missing_AudID]]

                            for SimilarAud in SimilarAudiences:

                                Sim_AudID = Name2ID[SimilarAud[0]]

                                if (Sim_AudID in UserAudienceMatrix.index):

                                    if (UserAudienceMatrix['TargetUserMean'].loc[Sim_AudID] != 'M'):

                                        if SimilarAud[2] == '+':
                                            UserAudienceMatrix['TargetUserMean'].loc[Missing_AudID] = UserAudienceMatrix['TargetUserMean'].loc[Sim_AudID]
                                            UserAudienceMatrix['TargetObsNumber'].loc[Missing_AudID] = 1
                                            break
                                        else:
                                            UserAudienceMatrix['TargetUserMean'].loc[Missing_AudID] = 1 - UserAudienceMatrix['TargetUserMean'].loc[Sim_AudID]
                                            UserAudienceMatrix['TargetObsNumber'].loc[Missing_AudID] = 1
                                            break
                        except:
                            continue



                    UserAudienceMatrix_Reduced = UserAudienceMatrix[(UserAudienceMatrix['SourceUserMean'] != 'M') & (UserAudienceMatrix['TargetUserMean'] != 'M')].copy()

                    WeighCorr = WeightedCorrelation(UserAudienceMatrix_Reduced['SourceUserMean'], UserAudienceMatrix_Reduced['TargetUserMean'], 
                                               UserAudienceMatrix_Reduced['SourceObsNumber'] * UserAudienceMatrix_Reduced['TargetObsNumber'])


                    if WeighCorr > SimilarUsers['SimilarityIndex'].min():

                        MinIndex = SimilarUsers['SimilarityIndex'].argmin()
                        SimilarUsers.loc[MinIndex] = Target_UserID, WeighCorr


        print(list(SimilarUsers.sort_values(by='SimilarityIndex', ascending=False)['UserID']))


        return None
    
    else:
        print('There is no enough data for Source User (ID: {})'.format(SourceUserID))
        print('Similarity analysis cannot be completed!')
        return None
    
    
        
        
if __name__ == '__main__':
    
    try:
        SourceUserID = sys.argv[1]
        GetSimilarUsers(SourceUserID)
    except:
        GetSimilarUsers()

