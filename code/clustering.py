import nltk
import pandas as pd
import numpy as np
import Processor as processor
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import random as sparse_random
import os.path
from os import path
#from numba import jit, cuda

  
if __name__ == '__main__':
  ##Import data
  df = pd.read_pickle("./data/loneliness_forum_data_page2_50.pkl")
  df1 = pd.read_pickle("./data/loneliness_forum_data_50_100.pkl")
  #df2 = pd.read_pickle("./data/loneliness_forum_data_100_150.pkl")

  df = df.append(df1)
  #df = df.append(df2)

  #print(df)
  print(len(df))

  #select posts without the replies
  #df_q_only = df['Posts']

  #df_questions_only = df['Posts'].copy()
  df_questions_only = pd.DataFrame()
  df_questions_only['Posts'] = df['Posts'].values
  df_questions_only = df_questions_only.drop_duplicates()
  print(len(df_questions_only))

  print(df_questions_only.head())

  ##TEST ONLY!!!: select sample 100 rows
  #df_questions_only = df_questions_only.sample(n=3000)

  print("Preparing text, len "+str(len(df_questions_only)))
  df_questions_only["PreparedText"] = df_questions_only['Posts'].apply(lambda x: processor.prepareText(x))
  df_questions_only["Tokens"] = df_questions_only['PreparedText'].apply(lambda x: nltk.word_tokenize(x))

  df = df_questions_only



  print("Extracting features")
  # TF-IDF using 2-gram
  tv = TfidfVectorizer(use_idf=True, min_df=0.0, max_df=1.0, ngram_range=(1,2))
  tfidf_2gram_features = tv.fit_transform(df['PreparedText'])
  print('TF-IDF features size:', str(tfidf_2gram_features.shape[0]), " x ", str(tfidf_2gram_features.shape[1]))

  #LDA
  print("LDA")
  # create a dictionary from the data
  #!pip install --upgrade --force-reinstall gensim
  import gensim
  from gensim import corpora
  dictionary = corpora.Dictionary(df["Tokens"])

  # convert to bag-of-words corpus
  corpus = [dictionary.doc2bow(text) for text in df["Tokens"]]

  # Compute Perplexity
  #print('\nPerplexity: ', ldamodel.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

  NUM_TOPICS = 1

  OPTIMIZE_USING_COHERENCE = False
  if OPTIMIZE_USING_COHERENCE:
    from gensim.models.coherencemodel import CoherenceModel
    preplexity_for_N = []
    coherence_for_N  = []
    Ns = []

    plotPerplexity = False

    for i in range(2,20):
      print ("calculating coherance for ", str(i), " topics")
      ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=i, id2word= dictionary, passes = 5)
      Ns.append(i)
      if plotPerplexity:
        preplexity_for_N.append(ldamodel.log_perplexity(corpus))

      # Compute Coherence Score
      coherence_model_lda = CoherenceModel(model=ldamodel, texts=df['Tokens'], dictionary=dictionary, coherence='c_v')
      coherence_lda = coherence_model_lda.get_coherence()
      coherence_for_N.append(coherence_lda)


    if plotPerplexity:
      #Plot preplexity
      fig, ax = plt.subplots()
      ax.plot(Ns, preplexity_for_N)
      ax.set(xlabel='Number of topics', ylabel='Preplexity',
            title='Forum data preplexity vs Number of topics')
      ax.grid()
      fig.savefig("PvsN.png")
      plt.show()

    #Plot coherance
    fig, ax = plt.subplots()
    ax.plot(Ns, coherence_for_N)
    ax.set(xlabel='Number of topics', ylabel='Preplexity',
          title='Forum data Coherence vs Number of topics')
    ax.grid()
    fig.savefig("COHvsN.png")
    plt.show()

    if plotPerplexity:
      min_value = min(preplexity_for_N) 
      min_index = preplexity_for_N.index(min_value) 
      print("NUM_TOPICS min P:", str(min_index))
      NUM_TOPICS = min_index

    max_value = max(coherence_for_N) 
    max_index = coherence_for_N.index(max_value) 
    print("NUM_TOPICS max C:", str(max_index))
    NUM_TOPICS = max_index

  NUM_TOPICS = 6 #based on local max of coherence

  ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=NUM_TOPICS, id2word= dictionary, passes = 5)
  print("LDA Topics:")
  print(ldamodel.print_topics())
  doc_lda = ldamodel[corpus]

  import pyLDAvis
  import pyLDAvis.gensim_models as gensimvis

  #pyLDAvis.enable_notebook()
  #vis = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)
  #vis
  print("Saving LDA_Visualization.html")
  visualisation = pyLDAvis.gensim_models.prepare(ldamodel, corpus, dictionary)
  pyLDAvis.save_html(visualisation, 'LDA_Visualization.html')

  #Word2Vec

  #pip  install  numba
  #To use the directive below you need to install cuda
  #https://developer.nvidia.com/cuda-90-download-archive
  # function optimized to run on gpu 
  #@jit(target ="cuda")
  # Word2Vec word embeddings
  def document_vectorizer(corpus, model, num_features):
      vocabulary = set(model.wv.index2word)
      def average_word_vectors(words, model, vocabulary, num_features):
          feature_vector = np.zeros((num_features,), dtype="float64")
          nwords = 0.
          for word in words:
              if word in vocabulary:
                  nwords = nwords + 1.
                  feature_vector = np.add(feature_vector, model.wv[word])
          if nwords:
              feature_vector = np.divide(feature_vector, nwords)
          return feature_vector
      features = [average_word_vectors(tokenized_sentence, model, vocabulary, num_features) for tokenized_sentence in corpus]
      return np.array(features)



  #Word2Vec
  import gensim
  # build word2vec model
  w2v_num_features = 1000
  w2v_model = gensim.models.Word2Vec(df['Tokens'], size=w2v_num_features, window=100, min_count=2, sample=1e-3, sg=1, iter=5, workers=10)
  # generate document level embeddings
  # generate averaged word vector features from word2vec model
  avg_wv_features = document_vectorizer(corpus=df['Tokens'], model=w2v_model, num_features=w2v_num_features)

  print("Word2Vec:" + str(avg_wv_features.shape))


  """### Create Dictionary of features
  We put data into disctionary to iterate through
  """
  features_dictionary = {
      'TF-IDF_2gram' :  tfidf_2gram_features,   #1-2 grams of words
      'W2V'          :  avg_wv_features,        #ginsim w2v  
  }

  N_ITER = 1000
  ##Visualizing data in 2D
  ## using TSNE or TruncatedSVD depending on parameter useTSNE
  ## When useTSNE = true the run takes MUCH longer but the results might be better
  def plot2D(features, title="", useTSNE=False, labelData=False, x_label="", y_label=""):
    if useTSNE:
      model = TSNE(n_components=2, random_state=42, n_iter = N_ITER)
      data_2D = model.fit_transform(features)
    else:
      svd = TruncatedSVD(n_components=2, random_state=42, n_iter = N_ITER)
      data_2D = svd.fit_transform(features)

    #no labels
    from itertools import repeat
    zero_df = pd.DataFrame.from_dict({'Zeros': list(repeat(0, len(df)))})
    y = zero_df['Zeros'].values

    data_2D = np.vstack((data_2D.T, y)).T
    df_2D = pd.DataFrame(data=data_2D, columns=("Dim_1", "Dim_2", "label"))

    fig, ax = plt.subplots()
    scatter = ax.scatter(df_2D["Dim_1"].values.tolist(), df_2D["Dim_2"].values.tolist(), s=4, cmap='viridis')
    # produce a legend with the unique colors from the scatter
    legend1 = ax.legend(*scatter.legend_elements(),
                      loc="best", title="Classes")
    ax.add_artist(legend1)
    ax.title.set_text(title)
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)

    plt.show()

  plotFeature = False
  if plotFeature:
    print("Plotting Features")  
    """### Plot the data"""
    for name, data in features_dictionary.items():
      plot_title = name + " features in 2D"
      plot2D(data, useTSNE=True, title=plot_title)


  """## Clustering
  #### Common functions
  """

  #common functions
  from sklearn.manifold import TSNE
  from sklearn.decomposition import TruncatedSVD
  from scipy.sparse import random as sparse_random

  N_ITER = 1000

  ##Plot result of clustering in 2D
  def plotClusteringData(features, title, clustered_data_label):
    model = TSNE(n_components=2, random_state=42, n_iter = N_ITER)
    data_2D = model.fit_transform(features)

    y = df[clustered_data_label].values
    data_2D = np.vstack((data_2D.T, y)).T
    df_2D = pd.DataFrame(data=data_2D, columns=("Dim_1", "Dim_2", "label"))


    fig, ax = plt.subplots()
    scatter = ax.scatter(df_2D["Dim_1"].values.tolist(), df_2D["Dim_2"].values.tolist(), s=4, c=y, cmap='viridis')
    # produce a legend with the unique colors from the scatter
    legend1 = ax.legend(*scatter.legend_elements(),
                      loc="best", title="Classes")
    ax.add_artist(legend1)
    ax.title.set_text(title)
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)

    plt.show()

  """### NUMBER OF CLUSTERS
  This takes ~30 min to run 
  """

  #Number of clusters
  from sklearn.cluster import KMeans
  MAX_ITER = 300
  N_INIT = 50
  NUM_CLUSTERS = NUM_TOPICS

  def plotElbow(data, title1, max_clusters=20):
    wcss = []
    for i in range(2, max_clusters):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
    
    fig, ax = plt.subplots()
    ax.plot(range(2, max_clusters), wcss)
    ax.set(xlabel='Number of clusters', ylabel='WCSS', title=title1)
    ax.grid()
    fig.savefig("Elbow"+title1+".png")
    plt.show()


    runElbow = False
    print("Plotting elbow")
    #TODO: don't forget to set to True before submitting
    if runElbow :
      for name, data in features_dictionary.items():
        plot_title = 'Elbow Method '+ name
        plotElbow(data, plot_title)


  #re-run only if filename does not exist
  print("Clustering")
  df_file_name = "./kmean_res_df.pkl"
  if path.exists(df_file_name):
    # load the data
    df = pd.read_pickle(df_file_name)
  else:

    for name, data in features_dictionary.items():
      title = 'kmean_'+ name
      print(title)
      km = KMeans(n_clusters=NUM_CLUSTERS, max_iter=MAX_ITER, n_init=N_INIT, random_state=42).fit(data)
      KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=MAX_ITER,
            n_clusters=NUM_CLUSTERS, n_init=N_INIT, n_jobs=None, precompute_distances="auto",
            random_state=42, tol=0.0001, verbose=0)
      #label 
      df[title] = km.labels_

    df.to_pickle(df_file_name)

  ##select random 1000 rows
  df_sample = df.sample(n=1000)
  df_sample.to_excel("Random1000_questions_KMean.xlsx", engine='xlsxwriter')

  for name, data in features_dictionary.items():
      plotClusteringData(data, "kmean_"+name,  'kmean_'+ name)


  """#### Plotting the clusters"""

  from sklearn.cluster import AgglomerativeClustering
  print("Cluster using AgglomerativeClustering")
  """#### Clustering the data"""
  model = AgglomerativeClustering(affinity='euclidean', linkage='ward')

  from sklearn.cluster import AgglomerativeClustering
  #re-run only if filename does not exist
  df_file_name = "./agglom_res_df.pkl"
  if path.exists(df_file_name):
    # load the data
    df = pd.read_pickle(df_file_name)
  else:

    for name, data in features_dictionary.items():
      title = 'agglom_'+ name
      print(title)
      #possible affinities (distance measures) “euclidean”, “l1”, “l2”, “manhattan”, “cosine”, or “precomputed”
      cluster = AgglomerativeClustering(n_clusters=NUM_CLUSTERS, affinity='euclidean', linkage='ward')
      if name == 'W2V' or name == 'TF-IDF_RD' or name == 'lda': #word to vec produces dense representation
        cluster.fit_predict(data)
      else:
        #other features in sparce representation
        cluster.fit_predict(data.toarray())
      #cluster labels 
      df[title] = cluster.labels_

    df.to_pickle(df_file_name)

  for name, data in features_dictionary.items():
      plotClusteringData(data, "Agglomerative clustering clustering of "+name,  'agglom_'+ name) 

