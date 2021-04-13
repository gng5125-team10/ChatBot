import pandas as pd
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from joblib import dump, load


df = pd.read_pickle("./kmean_res_df.pkl")

print(df.head())
print(len(df))

#sklearn.svm should be trained by calling TrainModel before use
model = make_pipeline(StandardScaler(with_mean=False), SVC(gamma='auto')) 

#TFIDF vectorizer, should be initialized by calling TrainModel
tfidf_vect = TfidfVectorizer(ngram_range=(1,2))

#feature TFIDF
features_tfidf = tfidf_vect.fit_transform(df["Posts"].values)

#Train SVM model
model.fit(features_tfidf, df["kmean_W2V"].values)
dump(model, 'SVM_TFIDF.model') 

X = df['Posts'].values
y = df['kmean_TF-IDF_2gram'].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state = 3)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state = 3)
tfidf_vect = TfidfVectorizer(ngram_range=(1,2))
X_train_tfidf = tfidf_vect.fit_transform(X_train)
X_test_tfidf = tfidf_vect.transform(X_test)

model.fit(X_train_tfidf, y_train)
tfidf_y_pred = model.predict(X_test_tfidf)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print(classification_report(tfidf_y_pred, y_test))
