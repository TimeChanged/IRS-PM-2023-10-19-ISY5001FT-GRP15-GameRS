from flask import Flask, render_template, request, redirect, session,jsonify
import pickle
import nltk
import random
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import pandas as pd

nltk.download('punkt')

app = Flask(__name__)
app.secret_key = 'abcdef'
app.static_folder = 'static'
USER_DB_FILE ='./users.txt'
# Load FAQ data
faq = pd.read_csv("Service.txt", delimiter="\t", header=None, names=["Q", "A"], encoding='iso-8859-1')
qns = faq["Q"]
answers = faq["A"]
TfidfVec = TfidfVectorizer()
tfidf = TfidfVec.fit_transform(qns)

# Load game recommendation data
new_df = pickle.load(open("./data/game_data.pkl", 'rb'))
vectors = pickle.load(open("./data/vectors.pkl", 'rb'))
cv = pickle.load(open("./data/cv.pkl", 'rb'))
similarity = pickle.load(open("./data/similarity.pkl", 'rb'))
game_image=pd.read_csv("./data/image_data.csv")

GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey")
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]

game_name_history = [] # Initialize an empty list to store game names
game_genre_history = []

def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

def response(user_response):
    robo_response = ''
    Q = ''
    A = ''
    new = TfidfVec.transform([user_response])  # vectorize the input to the same dimension space
    vals = cosine_similarity(new[0], tfidf)
    flat = vals.flatten()
    idx = flat.argsort()[-1]
    sim_max = flat[idx]
    if sim_max <= 0.2:
        robo_response = "I am sorry! I don't have an answer for that."
        return robo_response, Q, sim_max, A
    else:
        robo_response = "Similar question found!"
        Q = qns[idx]
        A = "Bill: " + answers[idx]
        return robo_response, Q, sim_max, A

def say(robo_response, Q, score, A):
    print(A)
    return A

def read_user_db():
    users = {}
    with open(USER_DB_FILE, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                username, password = line.split(':')
                users[username] = password
    return users

def write_user_db(users):
    with open(USER_DB_FILE, 'w') as file:
        for username, password in users.items():
            file.write(f'{username}:{password}\n')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        action = request.form['action']  

        users = read_user_db()

        if action == 'register':
            if username in users:
                return 'User already exists, please use another name'

            users[username] = password
            write_user_db(users)

            session['username'] = username
            return redirect('/home')
        elif action == 'login':
            if username in users and users[username] == password:
                session['username'] = username
                return redirect('/home')
            else:
                return redirect('/')
    return render_template('index.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect('/')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/user')
def user():
    return render_template('user.html')

@app.route('/Name_Genre')
def Name_Genre():
    return render_template('Name_Genre.html')

@app.route('/Bill')
def Bill():
    return render_template('Bill.html')

@app.route('/Ask')
def Ask():
    return render_template('Ask.html')

dialogue = []  
dialogue.append("Bill: My name is Bill. I am the Service bot for this recommendation system, if you have any questions, please feel free to ask. If you want to end the conversation, type Bye!")
@app.route('/chatbot', methods=['POST'])
def chatbot():
        user_input = request.form['user_input']
        user_input = user_input.lower()
        dialogue.append("User: " + user_input)
        if(user_input.lower() !='bye'):
            if(user_input=='thanks' or user_input=='thank you' ):
                dialogue.append("Bill: You are welcome..")
                return render_template('Chatbot.html', dialogue=dialogue)
            else:
                if(greeting(user_input)!=None):
                    dialogue.append("Bill: " + greeting(user_input))
                    return render_template('Chatbot.html', dialogue=dialogue)
                else:
                    print("Bill: ",end="")
                    Answer=say(*response(user_input))
                    dialogue.append("Bill: " + Answer)
                    return render_template('Chatbot.html', dialogue=dialogue)
        else:
            dialogue.append("Bill: Bye! take care...")
            return render_template('Chatbot.html', dialogue=dialogue)

@app.route('/recommend', methods=['POST'])
def recommend():
    game_name = request.form['game_name']
    index = new_df[new_df['Name'] == game_name].index[0]

    game_name_history.append(game_name)
    game_name_counts = pd.Series(game_name_history).value_counts()
    game_name_percentages = game_name_counts / game_name_counts.sum()

    plt.figure(figsize=(8, 6))
    game_name_counts.plot(kind='bar')
    plt.title('Game Inputs Count')
    plt.xlabel('Game Name')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.savefig('static/img/bar_chart.png')  # Save the bar chart as an image
    plt.close()  # Close the bar chart figure

    # Plot a pie chart of the game name percentages
    plt.figure(figsize=(8, 6))
    plt.pie(game_name_percentages, labels=game_name_percentages.index, autopct='%1.1f%%')
    plt.title('Game Inputs Percentage')
    plt.savefig('static/img/pie_chart.png')  # Save the pie chart as an image
    plt.close()  # Close the pie chart figure

    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_games = [(new_df.iloc[i[0]]['Name'],game_image.iloc[i[0]]['Headerimage']) for i in distances[1:6]]
    return render_template('recommend.html', game_name=game_name, recommended_games=recommended_games)

@app.route('/reply', methods=['GET'])
def search_games():
    input = request.args.get('input')
    games = new_df['Name']
    
    matches = []
    for game in games:
        if game.lower().startswith(input.lower()):
            matches.append(game)
    
    resp = jsonify(matches)
    resp.headers['Content-Type'] = 'application/json'
    return resp

@app.route('/information', methods=['POST'])
def information():
    game_name = request.form['game_name']
    index = new_df[new_df['Name'] == game_name].index[0]
    recommended_games = {
        'Name': new_df.loc[index, 'Name'],
        'About the game': new_df.loc[index, 'About the game'],
        'Headerimage': game_image.loc[index, 'Headerimage']
    }
    return render_template('information.html', game_name=game_name, recommended_games=recommended_games)

@app.route('/recommend_genre', methods=['POST'])
def recommend_genre():
    genre = request.form['genre']
    genre_vector = cv.transform([genre]).toarray()

    game_genre_history.append(genre)
    game_genre_counts = pd.Series(game_genre_history).value_counts()
    game_genre_percentages = game_genre_counts / game_genre_counts.sum()

    plt.figure(figsize=(8, 6))
    game_genre_counts.plot(kind='bar')
    plt.title('Game Inputs Count')
    plt.xlabel('Game Genre')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.savefig('static/img/Gbar_chart.png')  # Save the bar chart as an image
    plt.close()  # Close the bar chart figure

    # Plot a pie chart of the game name percentages
    plt.figure(figsize=(8, 6))
    plt.pie(game_genre_percentages, labels=game_genre_percentages.index, autopct='%1.1f%%')
    plt.title('Game Inputs Percentage')
    plt.savefig('static/img/Gpie_chart.png')  # Save the pie chart as an image
    plt.close()  # Close the pie chart figure

    distances = []
    for i in range(len(vectors)):
        a = vectors[i, :]
        a = a.reshape(1, -1)
        distance = euclidean_distances(genre_vector, a)
        distances.append((distance, i))
    distances = sorted(distances, key=lambda x: x[0])
    min_distances = distances[:5]
    min_indexes = [idx for _, idx in min_distances]
    recommended_games = [(new_df.iloc[idx]['Name'],game_image.iloc[idx]['Headerimage']) for idx in min_indexes]
    return render_template('recommend_genre.html', genre=genre, recommended_games=recommended_games)

if __name__ == '__main__':
    app.run(debug=True)