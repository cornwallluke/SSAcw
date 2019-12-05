import pandas as pd
import numpy as np
#from flask import Flask
from scipy.sparse.linalg import svds

books = pd.read_csv('gdb/clean.csv')
ratings = pd.read_csv('gdb/ratings.csv')


bookratings = pd.merge(books, ratings, on="book_id")#.astype({'rating':'float32'})
userpiv=bookratings.pivot_table(index="user_id",columns="book_id",values="rating").fillna(0)
#print(userpiv.head())
userpiv=userpiv.sub(userpiv.mean(axis=1),axis=0)
npuserpiv=userpiv.values
print("yeet")
#np.linalg.svd(userpiv)
u, sigma, vt = svds(userpiv,k=2)
sigma = np.diag(sigma)
#print(u)
#print(sigma)
v=vt.T
print(userpiv)
print(v[0])
#print(np.concatenate(userpiv.index.values,((v-v[0])**2).sum(axis=1)))

