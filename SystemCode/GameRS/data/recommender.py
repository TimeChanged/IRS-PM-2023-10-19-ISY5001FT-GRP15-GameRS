import pandas as pd
import pickle
games_data=pd.read_csv('./game.csv')
games_data.head()
games_data=games_data[['AppID','Name','About the game','Categories','Genres','Tags']]

new_games_data=pd.DataFrame(games_data)

new_games_data['Genres']=new_games_data['Genres'].apply(lambda x:x.split())
new_games_data['Categories']=new_games_data['Categories'].apply(lambda x:x.split())
new_games_data['Tags']=new_games_data['Tags'].apply(lambda x:x.split())
new_games_data['About the game']=new_games_data['About the game'].apply(lambda x:x.split())

new_games_data['Categories']=new_games_data['Categories'].apply(lambda x:[i.replace(" ","") for i in x])
new_games_data['Genres']=new_games_data['Genres'].apply(lambda x:[i.replace(" ","") for i in x])
new_games_data['Tags']=new_games_data['Tags'].apply(lambda x:[i.replace(" ","") for i in x])

new_games_data['tags']=new_games_data['About the game'] + new_games_data['Categories'] + new_games_data['Genres'] + new_games_data['Tags']

new_df=new_games_data[['AppID','Name','tags']]

new_df['tags']=new_df['tags'].apply(lambda x:" ".join(x))
new_df['tags']=new_df['tags'].apply(lambda x:x.lower())

from nltk.stem.porter import PorterStemmer
ps= PorterStemmer()
def stem(text):
  y=[]
  for i in text.split():
    y.append(ps.stem(i))
  return " ".join(y)
new_df['tags']=new_df['tags'].apply(stem)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000,stop_words='english')
vectors=cv.fit_transform(new_df['tags']).toarray()
print(vectors.shape)

from sklearn.metrics.pairwise import cosine_similarity
similarity=cosine_similarity(vectors)
def recommend(game):
    index = new_df[new_df['Name'] == game].index[0]
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
    for i in distances[1:6]:
        print(new_df.iloc[i[0]].Name)
print(similarity.shape)
first_row = similarity[0]
print(first_row)
save_path = './similarity.pkl'
pickle.dump(similarity, open(save_path, 'wb'))
