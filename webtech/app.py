from flask import Flask, render_template, request, session, jsonify
import pandas as pd
import json
yeet = 'yeet'
app = Flask(__name__)



app.secret_key = "ganggang"
locale = {}
with open('locales.json','r') as locales:
    locale = json.loads(locales.read())

books = pd.read_csv('gdb/clean.csv')#.set_index('title')
ratings = pd.read_csv("gdb/ratings.csv")

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
    session['username'] = request.form["username"]
    # session['locale'] = 
    # print(session.get('username'))
    return '200'

def getnewuserid():
    ratings = pd.read_csv("gdb/ratings.csv")
    print(ratings['user_id'].max())

@app.route("/adduser",methods = ['POST'])
def adduser():
    
    # print(request.form)
    users = pd.read_csv('users.csv')
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
    users.to_csv("users.csv",index = False)
    return "200"
@app.route("/updaterating",methods = ["POST"])
def updaterating():
    ratings = pd.read_csv("gdb/ratings.csv")
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
    ratings.to_csv('gdb/ratings.csv',index = False)
    return "200"
@app.route("/edituser",methods = ['POST'])
def edituser():
    
    print(request.form)
    users = pd.read_csv('users.csv').set_index('username')
    old = dict(users.loc[session['username']])
    users = users.drop(session['username'])
    users = users.append({
        "username":session['username'],
        "password":request.form['password'],
        "fname":request.form['fname'],
        "sname":request.form['sname'],
        "locale":request.form['locale'],
        "uid":80
    },ignore_index=True)
    users.to_csv("users.csv",index = False)
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

@app.route("/top24",methods = ["GET"])
def gettop24():
    ratings = pd.read_csv("gdb/ratings.csv")

    users = pd.read_csv('users.csv').set_index('username')
    user = dict(users.loc[session['username']])
    uid = user['uid']
    # a = pd.merge(books,ratings.loc[(ratings['user_id']==uid)],on = "book_id", how = "left")
    retr = pd.merge(books,ratings.loc[(ratings['user_id']==uid)],on = "book_id", how = "left")[:24].fillna("null").to_dict('record')
    print(retr)
    print("\n\n")
    print(jsonify(books[:24].to_dict('record')))
    print(books)
    return jsonify(retr)
    # return jsonify(books[:24].to_dict('record'))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)