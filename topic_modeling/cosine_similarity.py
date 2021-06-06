'''
Similaridade por Cosseno
---------------------------------------------------
- Coletar os dados filtrados (remoção de stopwords e lematização);
- Utilizar a bibliotéca scikit-learn para;
- Calcular o TF-IDF (Term Frequency - Inverse Document Frequency);
- Utilizar a matriz TF-IDF para calcular a similaridade por cosseno de todo o corpus;
- Salvar isso em matrizes esparsas.

'''

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
path = "../files/"

#Leitura do dataset filtrado
df = pd.read_csv( path + "4mula_filtered.csv" )
#Coleta do corpus filtado
corpus = df['filtered_lyrics'].tolist()
#Cálculo do TF-IDF
tf_idf = TfidfVectorizer(dtype=np.float32).fit_transform(corpus)

from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import save_npz, load_npz

#Batch sendo a quantidade de músicas dentro de cada arquivo
batch = 10000
filename = 'cosine_similarity/cos_sim'
for i in range(4):
    #csim - Uma Matriz esparsa comparando 10000 músicas (a cada arquivo) com todas do corpus
    csim = cosine_similarity(tf_idf[i*batch:(i+1)*batch],tf_idf,dense_output=False)
    save_npz(filename + str(i+1) + '.npz',csim)
#Último arquivo 'cos_sim5.npz' guarda as ultimas 10572 músicas restantes
csim = cosine_similarity(tf_idf[40000:],tf_idf,dense_output=False)
save_npz(filename + '5.npz',csim)