from flask import Flask, request
import subprocess

app = Flask(__name__)


@app.route('/', methods=['GET'])
def g():
    return "hiiiii"


@app.route('/api/query', methods=['POST'])
def get_query_from_react():
    data = request.get_json()
    actor2Name = data['data']['actor_2_name']
    actor1Name = data['data']['actor_1_name']
    actor3Name = data['data']['actor_3_name']
    directorName = data['data']['director_name']
    country = data['data']['country']
    cr = data['data']['cr']
    language = data['data']['language']
    actor1Likes = data['data']['actor_1_likes']
    actor2FacebookLikes = data['data']['actor_2_facebook_likes']
    actor3FacebookLikes = data['data']['actor_3_facebook_likes']
    directorFacebookLikes = data['data']['director_facebook_likes']
    castTotalFacebookLikes = data['data']['cast_total_facebook_likes']
    budget = data['data']['budget']
    gross = data['data']['gross']
    genres = data['data']['genres']
    imdbScore = data['data']['imdb_score']

    print(genres)

    result = subprocess.run(['python', 'Model.py', actor1Name, actor2Name, actor3Name, directorName, country, cr, language, actor1Likes, actor2FacebookLikes, actor3FacebookLikes, directorFacebookLikes, castTotalFacebookLikes, budget, gross, genres, imdbScore],
                            )

    return data
