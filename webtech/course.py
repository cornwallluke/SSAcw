import pandas as pd
import numpy as np
from flask import Flask

# app = Flask(__name__)

# @app.route("/output")
# def output():
# 	return "Hello World!"

# if __name__=="__main__":
#     app.run()
books = pd.read_csv('gdb/books.csv')
booktags = pd.read_csv('gdb/book_tags.csv')
ratings = pd.read_csv('gdb/ratings.csv')
tags = pd.read_csv('gdb/tags.csv')


genres = [i.lower() for i in ["adventure",	"Art", "Alternate history", "Autobiography", "Anthology", "Biography", "Chick lit",	"Book-review", "Children's", "Cookbook", "Comic book", "Diary", "Dictionary", "Crime", "Encyclopedia", "Drama", "Guide", "Fairytale", "Health", "Fantasy", "History", "Graphic novel", "Journal", "Historical fiction", "Math", "Horror", "Memoir", "Mystery", "Prayer", "Paranormal-romance", "Religion", "Picture book", "Textbook", "Poetry", "Review", "Political thriller", "Science", "Romance", "Self help", "Satire", "Travel", "Science fiction", "True crime", "Short story", "Suspense", "Thriller", "Young adult"]]
tags=tags[tags['tag_name'].isin(genres)]
#yeet=yeet[yeet['tag_name'].isin(genres)]

#print(books)
yeet = pd.merge(books,booktags,on='goodreads_book_id')
yeet = pd.merge(yeet, tags, on='tag_id')
yeet=yeet.drop(columns=["goodreads_book_id", "best_book_id", "work_id", "books_count", "isbn", "isbn13", "authors", "original_publication_year", "original_title", "language_code", "average_rating", "ratings_count", "work_ratings_count", "work_text_reviews_count", "ratings_1", "ratings_2", "ratings_3", "ratings_4", "ratings_5", "image_url", "small_image_url"])


#print(yeet.head())
yeet=yeet[yeet.groupby('title',sort=False)['count'].transform(max)==yeet['count']].drop_duplicates(subset='book_id')
yeet.to_csv('gdb/clean.csv')
#print(yeet)

bookratings = pd.merge(yeet, ratings, on="book_id")
userpiv=bookratings.pivot_table(index="user_id",columns="book_id",values="rating").fillna(0)
#print(userpiv.head())
userpiv=userpiv.sub(userpiv.mean(axis=1),axis=0)
print(userpiv.head())
# tonump=userpiv.values()

#print(movieratings.head())