from flask import Flask, render_template, request, session, jsonify
import pandas as pd
import json
import numpy as np
from scipy.sparse.linalg import svds
yeet = 'yeet'
app = Flask(__name__)

class recommender:
    def __init__(self):
        self.books = pd.read_csv('gdb/cleaned.csv')#.set_index('title')
        self.ratings = pd.read_csv("gdb/ratings2.csv")

        self.bookratings = pd.merge(self.books, self.ratings, on="book_id")#.astype({'rating':'float32'})
        self.userpiv=self.bookratings.pivot_table(columns="book_id",index="user_id",values="rating")
        #print(userpiv.head())
        self.userpiv=self.userpiv.sub(self.userpiv.mean(axis=1,skipna = True),axis=0).fillna(-0.05)
        self.npuserpiv=self.userpiv.values
        # print(userpiv.shape)
        self.u, self.sigma, self.vt = svds(self.userpiv,k=40)
        self.sigma = np.diag(self.sigma)
        self.vtsig = self.sigma @ self.vt
        self.v = self.vtsig.T
    def update(self):
        self.books = pd.read_csv('gdb/cleaned.csv')#.set_index('title')
        self.ratings = pd.read_csv("gdb/ratings2.csv")

        self.bookratings = pd.merge(self.books, self.ratings, on="book_id")#.astype({'rating':'float32'})
        self.userpiv=self.bookratings.pivot_table(columns="book_id",index="user_id",values="rating")
        #print(userpiv.head())
        self.userpiv=self.userpiv.sub(self.userpiv.mean(axis=1,skipna = True),axis=0).fillna(-0.05)
        self.npuserpiv=self.userpiv.values
        # print(userpiv.shape)
        self.u, self.sigma, self.vt = svds(self.userpiv,k=40)
        self.sigma = np.diag(self.sigma)
        self.vtsig = self.sigma @ self.vt
        self.v = self.vtsig.T
    def getrecs(self,user_ratings):
        user = user_ratings @ self.v

        disttobooks = np.sum(abs(self.v-user), axis = 1).reshape(-1,1)
        # print(disttobooks.shape)
        nearest = np.concatenate([np.asarray([self.userpiv.columns]).T,disttobooks],axis = 1)
        # print(nearest)
        # print(nearest[nearest[:,1].argsort()])
        nearest = nearest[nearest[:,1].argsort()]
        return pd.DataFrame(nearest).rename(columns = {0:'book_id', 1:'distance'}).astype({'book_id':'int'})[:24]

reco = recommender()
app.secret_key = "ganggang"
locale = {}
with open('locales.json','r') as locales:
    locale = json.loads(locales.read())

books = pd.read_csv('gdb/cleaned.csv')#.set_index('title')
ratings = pd.read_csv("gdb/ratings2.csv")

@app.route("/")
def root():
    # print(checklog())
    session['locale'] = 'english'
    if checklog():
            users = pd.read_csv('users.csv').set_index('username')
            old = dict(users.loc[session['username']])
            session['locale'] = old['locale']
            return render_template("index.html",locale = locale,user = old)
    return render_template("login.html",locale = locale)


@app.route("/logout",methods = ["POST"])
def logout():
    session.pop('username')
    return "200"

    
@app.route("/log",methods = ['POST'])
def login():
    # print(request.form)
    try:

        users = pd.read_csv('users.csv').set_index('username')
        un = request.form['username']
        old = dict(users.loc[un])
        print(old)
        print(request.form['password'])
        if old['password'] == request.form['password']:
            session['username'] = request.form["username"]

            session['locale'] = old['locale']

        print(session.get('username'))
        return '200'
    except:
        return '403'
def getnewuserid():
    ratings = pd.read_csv("gdb/ratings2.csv")
    
    print(ratings['user_id'].max())

@app.route("/adduser",methods = ['POST'])
def adduser():
    
    # print(request.form)
    users = pd.read_csv('users.csv')
    ratings = pd.read_csv("gdb/ratings2.csv")
    if request.form['username'] in users.username.values:
        return "user already exists"
    users = users.append({
        "username":request.form['username'],
        "password":request.form['password'],
        "fname":request.form['fname'],
        "sname":request.form['sname'],
        "locale":"english",
        "uid":max(ratings['user_id'].max(),users['uid'].max())+1
    },ignore_index=True)
    ratings = pd.read_csv("gdb/ratings2.csv")
    ratings.append({
        "user_id":max(ratings['user_id'].max(),users['uid'].max())+1,
        "book_id":99999999999999,
        "rating":0
    },ignore_index= True)
    ratings.to_csv('gdb/ratings2.csv',index = False)
    users.to_csv("users.csv",index = False)
    reco.update()
    return "200"
@app.route("/updaterating",methods = ["POST"])
def updaterating():
    ratings = pd.read_csv("gdb/ratings2.csv")
    # print(ratings['user_id'].max()+1)
    # print(request.values['rating'],request.values['title'])
    users = pd.read_csv('users.csv').set_index('username')
    user = dict(users.loc[session['username']])
    uid = user['uid']
    book = books.loc[books['title'] == request.values['title']].to_dict('record')[0]
    if len(ratings.loc[(ratings['user_id']==uid) & (ratings['book_id'] == book['book_id'])])>0:
        ratings.loc[(ratings['user_id']==uid) & (ratings['book_id'] == book['book_id']),'rating'] = request.values['rating']
    else:
        ratings = ratings.append({
            "user_id":uid,
            "book_id":book['book_id'],
            "rating":request.values['rating']
        },ignore_index= True)
    # print(ratings.loc[(ratings['user_id']==uid) & (ratings['book_id'] == book['book_id'])])
    ratings.to_csv('gdb/ratings2.csv',index = False)
    return "200"
@app.route("/edituser",methods = ['POST'])
def edituser():
    
    # print(request.form)
    users = pd.read_csv('users.csv').set_index('username')
    old = dict(users.loc[session['username']])
    users = users.drop(session['username'])
    users = users.append(pd.DataFrame([[
        request.form['fname'],
        request.form['locale'],
        request.form['password'],
        request.form['sname'],
        old['uid'],
        session['username']
    ]],columns = ["fname","locale", "password",  "sname", "uid","username"]),sort = False)#
    users.to_csv("users.csv", index = False)
    return "200"

@app.route("/setlocale",methods = ['POST'])
def setlocale():
    session['locale'] = request.form['locale']
# request.accept languages best match
def checklog():
    users = pd.read_csv('users.csv')
    if session.get('username') in users.username.values:
        return True
    return False

@app.route("/search", methods = ["GET"])
def search():
    
    return jsonify(books[books.title.str.contains(request.values['query'],case = False)][:10].to_dict("records"))

@app.route("/rated",methods = ["GET"])
def getrated():
    ratings = pd.read_csv("gdb/ratings2.csv")

    users = pd.read_csv('users.csv').set_index('username')
    user = dict(users.loc[session.get('username')])
    uid = user['uid']
    # a = pd.merge(books,ratings.loc[(ratings['user_id']==uid)],on = "book_id", how = "left")
    retr = pd.merge(books,ratings.loc[(ratings['user_id']==uid)],on = "book_id").fillna("null").to_dict('record')
    # print(retr)
    # print("\n\n")
    # print(jsonify(books[:24].to_dict('record')))
    # print(books)
    return jsonify(retr)

books = pd.read_csv('gdb/cleaned.csv')#.set_index('title')
ratings = pd.read_csv("gdb/ratings2.csv")

# bookratings = pd.merge(books, ratings, on="book_id")#.astype({'rating':'float32'})
# userpiv=bookratings.pivot_table(columns="book_id",index="user_id",values="rating")
# #print(userpiv.head())
# userpiv=userpiv.sub(userpiv.mean(axis=1,skipna = True),axis=0).fillna(0.05)
# npuserpiv=userpiv.values
# # print(userpiv.shape)
# u, sigma, vt = svds(userpiv,k=40)
# sigma = np.diag(sigma)
# vtsig = sigma @ vt
# v = vtsig.T
    
@app.route("/top24",methods = ["GET"])
def gettop24():
    # try:
    users = pd.read_csv('users.csv').set_index('username')
    user = dict(users.loc[session.get('username')])
    uid = user['uid']

    # books = pd.read_csv('gdb/cleaned.csv')#.set_index('title')
    ratings = pd.read_csv("gdb/ratings2.csv")

    # bookratings = pd.merge(books, ratings, on="book_id")#.astype({'rating':'float32'})
    # userpiv=bookratings.pivot_table(columns="book_id",index="user_id",values="rating")
    # #print(userpiv.head())
    # userpiv=userpiv.sub(userpiv.mean(axis=1,skipna = True),axis=0).fillna(0.05)
    # # npuserpiv=userpiv.values
    # print(userpiv.shape)
    # u, sigma, vt = svds(userpiv,k=60)
    # sigma = np.diag(sigma)
    # vtsig = sigma @ vt
    # v = vtsig.T
    user = pd.merge(ratings.loc[ratings['user_id'] == uid],books.loc[books['book_id'].isin(ratings['book_id'])], on = 'book_id', how = 'right').sort_values('book_id')['rating'].fillna(0).values
    # user = userpiv.T[uid] @ v

    # disttobooks = np.sum(abs(v-user), axis = 1).reshape(-1,1)
    # # print(disttobooks.shape)
    # nearest = np.concatenate([np.asarray([userpiv.columns]).T,disttobooks],axis = 1)
    # # print(nearest)
    # # print(nearest[nearest[:,1].argsort()])
    # nearest = nearest[nearest[:,1].argsort()]
    # reccs = pd.DataFrame(nearest).rename(columns = {0:'book_id', 1:'distance'}).astype({'book_id':'int'})[:24]
    reccs = reco.getrecs(user)
    return jsonify(pd.merge(books,reccs, on = "book_id").to_dict('record'))
    # except:
    #     return getrated()
    # return jsonify(books[:24].to_dict('record'))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)