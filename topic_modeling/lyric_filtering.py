''' 
***Filtragem das Letras***
---------------------------------------------------
1. Coleta do *dataset* [4Mula](https://github.com/4mulaDataset/4mula) e do arquivo de [*stopwords*](https://gist.github.com/alopes/5358189)
2. Filtragem dos campos da tabela irrelevantes para a aplicação
3. Formatação das letras
4. Utilização da bibliotéca [NLTK](https://www.nltk.org/) para *tokenização* das palavras
5. Filtragem das stopwords
6. Utilização da bibliotéca [Spacy](https://spacy.io/) para a lematização dos termos
7. Anexação das letras filtradas à coluna `filtered lyrics`
'''

import pandas as pd
import string
from nltk.tokenize import word_tokenize
import spacy
nlp = spacy.load("pt_core_news_md")

path = "../files/"

f = open(path + "stopwords.txt" ,"r")
_stopwords = f.read()
f.close()
stopwords = set()
for word in _stopwords.split("\n"):
    stopwords.add(word.rstrip())

df = pd.read_parquet( path + "4mula_metadata.parquet" )
m = df[df['music_lang'] == 'pt-br']

m = m.drop(columns=['music_id', 'art_id','music_lang','art_rank','related_art',
                    'related_music','main_genre', 'related_genre'])
m = m.reset_index(drop=True)

for i, row in m.iterrows():
    row['music_lyrics'] = row['music_lyrics'].replace('\\n',' ')
    row['music_lyrics'] = row['music_lyrics'].replace('  ',' ')


filtered_lyrics = []
for i, row in m.iterrows():
    doc = row['music_lyrics'].translate(str.maketrans('','',string.punctuation))
    words = word_tokenize(doc.lower())
    filtered = [word for word in words if not word in stopwords]            
    filtered_lyrics.append(filtered)

lemmatized_lyrics = []
for i, row in m.iterrows():
    doc = ' '.join(filtered_lyrics[i])
    filtered=""
    for token in nlp(doc):
        filtered += token.lemma_ + ' '
    lemmatized_lyrics.append(filtered[:-1])

m['filtered_lyrics'] = lemmatized_lyrics
m = m[['art_name','music_name','music_lyrics','filtered_lyrics']]
m.to_csv(path + '4mula_filtered.csv',index=False)