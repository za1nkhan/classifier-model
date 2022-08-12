import pickle
import json
import uvicorn
import requests
import re
#from sklearn.externals 
#import joblib 
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
#vectorizer = CountVectorizer()
import numpy as np
from fastapi import FastAPI

app = FastAPI()


model = pickle.load( open("mnbmodel.pkl","rb"))
vector = pickle.load(open("vector.pkl", "rb"))


#with open('improvedmulticlassmodel.pkl', 'rb') as f:
#   model = pickle.load(f)


def preprocessor(text):
    text = re.sub('<[^>]*>', '', text) # Effectively removes HTML markup tags
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    return text

#def bagofwords(text):
#    vectorizer.fit_transform(featnames)
#    #docs = np.array([text])
#    bag = vectorizer.transform([text])#docs)
#    text =  bag.toarray()
#    return text

def classify_event_category(model, tweet):
    #tweet = preprocessor(tweet)
    #tweet = bagofwords(tweet)
    #category = model.predict([tweet])[0]
    #category_prob = model.predict_proba([tweet])
    #return {'category': category, 'category_probability': category_prob[0][1]}
    data = [tweet]
    vect = vector.transform(data).toarray()
    result = model.predict(vect)
    proba = model.predict_proba(vect)
    print('prediction :', result)
    return result, proba; 

#def get_value(jsonpredict):
 #   if jsonpredict == 'ND':
  #      comma = 1
   #     return comma

@app.get('/get_event_category/')
async def get_event_category(tweet: str):
    eventCategory = ['ND', 'Traffic', 'Plc', 'Hlt', 'Opi', 'Spt', 'tech', 'life','travel', 'weather', 'Mny']
    prediction, proba = classify_event_category(model, tweet)
    jsonpredict = json.dumps(prediction.tolist())
    newpredict = jsonpredict.replace('[','').replace(']','')
    newpredict = int(newpredict)
    jsonproba = json.dumps(proba.tolist())
    jsonproba = jsonproba.replace('[','').replace(']','').split(',')
    print(jsonproba)
    #jsonpredict = jsonpredict.replace('[0]','ND').replace('[1]','Traffic').replace('[2]','Plc').replace('[3]','Hlt').replace('[4]','Opi').replace('[5]','Spt').replace('[6]','tech').replace('[7]','life').replace('[8]','travel').replace('[9]','weather').replace('[10]','Mny')
    jsonpredict = eventCategory[newpredict]
    jsonproba = (jsonproba[newpredict]).strip()
    jsonproba = float(jsonproba)
    return {'tweet':tweet, 'prediction': jsonpredict, 'probability': jsonproba}
    #{'message': 'Welcome to the spam detection API'}



@app.get('/')
def get_root():
    return {'message': 'Welcome to the spam detection API'}


if __name__ == '__main__':
    #text =  preprocessor('ABQ continues Flood Watch for East Slopes Sangre de Cristo Mountains, Jemez Mountains, South Central Mountains, Southern Sangre de Cristo Mountains, Southwest Mountains [NM] till Aug 4, 12:00 AM MDT')
    #classify_event_category(model, text)
    uvicorn.run(app,host="127.0.0.1",port=8002)