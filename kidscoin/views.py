from django.shortcuts import render
# views.py
from django.views.decorators.csrf import csrf_exempt
from rest_framework import generics
from rest_framework.parsers import JSONParser
from .modelss.reels import Reels
from .serializers import ReelSerializer
from django.http.response import JsonResponse
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import sklearn.linear_model
import sklearn
import pickle
import numpy as np
import json
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import scale
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.feature_extraction import text
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import scipy.sparse as sp
import pickle
@csrf_exempt
def ReelApi(request,id=0):
    if request.method=='GET':
        Reel = Reels.objects.all()
        for reel in Reel:
            print(reel.caption)
        Reel_serializer=ReelSerializer(Reel,many=True)
        return JsonResponse(Reel_serializer.data,safe=False)
    elif request.method=='POST':
        Reel_data=JSONParser().parse(request)
        #Reel_serializer=ReelSerializer(data=Reel_data)
        #if Reel_serializer.is_valid():
            #Reel_serializer.save()
        data=addtorecommendationdata(Reel_data)
        return JsonResponse(data,safe=False)
       
    elif request.method=='PUT':
        Reel_data=JSONParser().parse(request)
        Reel=Reels.objects.get(ReelId=Reel_data['ReelId'])
        Reel_serializer=ReelSerializer(Reel,data=Reel_data)
        if Reel_serializer.is_valid():
            Reel_serializer.save()
            return JsonResponse("Updated Successfully",safe=False)
        return JsonResponse("Failed to Update")
    elif request.method=='DELETE':
        Reel=Reels.objects.get(ReelId=0)
        Reel.delete()
        return JsonResponse("Deleted Successfully",safe=False)
def recommendreelapi(request,id=0):
    if request.method=='POST':
        
        body_data = JSONParser().parse(request)

    # Assuming the JSON data has a 'hashtag' attribute
        hashtags = body_data.get('hashtags', '')
        print("haaashtag", hashtags)
       
        hashtags_set = {hashtags}
        similar_rows = recommend_rows(hashtags_set)
        #print(similar_rows)
        #similar_rows['Date'] = similar_rows['Date'].astype(str)
        data_dict = similar_rows.to_json(orient='records')
        data_dict = json.loads(data_dict)
# Serialize dictionary to JSON
        
        #json_data = json.dumps(data_dict)
        return JsonResponse(data_dict,safe=False)
    
    elif request.method=='DELETE':
        Reel=Reels.objects.get(ReelId=0)
        Reel.delete()
        return JsonResponse("Deleted Successfully",safe=False)
def addalike(request,id=0):
    if request.method=='POST':
        
        body_data = JSONParser().parse(request)

    # Assuming the JSON data has a 'hashtag' attributee
        id = body_data.get('id', '')
        print(id)
       
        Id = id
        updatedrow = addtoliketodata(Id)
       
        return JsonResponse("Updated successfully",safe=False)
def recommend():
    tfid_matrix = None
    df= None
    with open('./tfidf_matrix.pkl', 'rb') as f:
         tfid_matrix = pickle.load(f)
    with open('./recommendersystem1Data.pkl', 'rb') as f:
         df = pickle.load(f)
    

def recommend_rows(hashtags_set):
    tfid_matrix = None
    df= None
    with open('tfidf_matrix.pkl', 'rb') as f:
         uni_tfidf, tfidf_matrix = pickle.load(f)
    with open('recommendersystem1Data.pkl', 'rb') as f:
         df = pickle.load(f)
         
    # Convert the set of hashtags into a single string
    hashtags_string = " ".join(hashtags_set)
    '''
    reel_data = {
    'Id': 150,
    'Date': pd.to_datetime("2024-04-10"),
    
    'likeCount': 500,
    'Caption': " usa at night",
    'Hashtags': "   #city  #girls",
    
    # Add more fields as needed
    }
    #df = df.append(reel_data, ignore_index=True)
    all_columns = df.columns.tolist()
    reel_data["Date_diff"] = 0
    
    df = df._append(reel_data, ignore_index=True)
    '''
    df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
    # Calculate date differences in days from the current date
    df["Date_diff"] = (pd.Timestamp.now() - df["Date"]).dt.days
    # Transform the hashtags string into TF-IDF vector
   
    #hashtags_test = " ".join("#beautiful  #city  #girls")
    #uni_tfidf.fit([hashtags_string])
    #new_tfidf_matrix = uni_tfidf.transform([hashtags_test])
    #captions = df["Hashtags"].tolist()
    #uni_tfidf = text.TfidfVectorizer(stop_words="english")
    #uni_tfidf.fit(captions)
    #new_tfidf_matrix = uni_tfidf.transform([hashtags_test])
# Concatenate the old and new TF-IDF matrices
    #combined_tfidf_matrix = sp.vstack([tfidf_matrix, new_tfidf_matrix])

    tfidf_vector = uni_tfidf.transform([hashtags_string])

    # Compute cosine similarity between the input vector and all rows
    similarities = cosine_similarity(tfidf_vector, tfidf_matrix)

    # Get indices of similar rows sorted by similarity scores
    similar_rows_indices = similarities.argsort()[0][-10:][::-1]

    # Return DataFrame rows corresponding to similar rows
    similar_rows = df.iloc[similar_rows_indices].copy()

    similar_rows['Similarity Score'] = similarities[0][similar_rows_indices] + ((df["likeCount"].iloc[similar_rows_indices].values / df["Date_diff"].iloc[similar_rows_indices].values)/1000)
    #similar_rows['Similarity Score'] = similarities[0][similar_rows_indices] - df["Date_diff"].iloc[similar_rows_indices].values
    similar_rows = similar_rows.sort_values(by='Similarity Score', ascending=False)
    print(similar_rows)
    return similar_rows
def addtorecommendationdata(data1):
    tfid_matrix = None
    df= None
    with open('tfidf_matrix.pkl', 'rb') as f:
         uni_tfidf, tfidf_matrix = pickle.load(f)
    with open('recommendersystem1Data.pkl', 'rb') as f:
         df = pickle.load(f)
    
    data = data1['reelDescription']
    index = data.find('#')

    if index != -1:
      data1['reelDescription'] = data[index:].strip()
      reel_data = {
      'Id': data1['_id'],
      'Date': pd.to_datetime("2024-04-10"),
    
      'likeCount': 500,
      'Caption': data1['reelDescription'],
      'Hashtags': data1['reelDescription'],
      
    # Add more fields as needed
    }
      reel_data["Date_diff"] = 0
    
      df = df._append(reel_data, ignore_index=True)
      hashtags_test = " ".join("#beautiful  #city  #girls")
      #uni_tfidf.fit([hashtags_string])
      #new_tfidf_matrix = uni_tfidf.transform([hashtags_test])
      captions = df["Hashtags"].tolist()
      uni_tfidf = text.TfidfVectorizer(stop_words="english")
      uni_tfidf.fit(captions)
      tfidf_matrix = uni_tfidf.transform(captions)
      pickle.dump(df, open('recommendersystem1Data.pkl', 'wb'))
      pickle.dump((uni_tfidf, tfidf_matrix), open("tfidf_matrix.pkl", 'wb'))
      return reel_data
    else:
      return "No '#' symbol found in reelDescription."
    
def addtoliketodata(data1):
    
    df= None
    
    with open('recommendersystem1Data.pkl', 'rb') as f:
         df = pickle.load(f)
    target_id = "6634ce7be63f2a7eef828ccd"
    row_index = df[df['Id'] == target_id].index
    print(row_index)
    if not row_index.empty:  # Ensure only one row founddd
       row_index = row_index[0]
    # 3. Update the 'likecount' column
       new_like_count = 100  # Replaceee 100 with your desiired like countt
       df.at[row_index, 'likeCount'] += new_like_count
       print("Like count updated successfully.")
       print(df.iloc[row_index])
    else:
       print("Error: ID not found or multiple rows found.")
      
   
    pickle.dump(df, open('recommendersystem1Data.pkl', 'wb'))
     
    return "done"
   
    
       

