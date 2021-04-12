import pandas as pd
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from joblib import dump, load


df = pd.read_pickle("./kmean_res_df.pkl")

print(df.head())

#sklearn.svm should be trained by calling TrainModel before use
model = make_pipeline(StandardScaler(with_mean=False), SVC(gamma='auto')) 

#TFIDF vectorizer, should be initialized by calling TrainModel
tfidf_vect = TfidfVectorizer(ngram_range=(1,2))

#feature TFIDF
features_tfidf = tfidf_vect.fit_transform(df["Posts"].values)

#Train SVM model
model.fit(features_tfidf, df["kmean_W2V"].values)
dump(model, 'SVM_TFIDF.model') 
