import pandas as pd
import numpy as np
#from flask import Flask
from scipy.sparse.linalg import svds
from time import time as ti
from yeet import *



books = pd.read_csv('gdb/clean.csv')
ratings = pd.read_csv('gdb/ratings.csv')


bookratings = pd.merge(books, ratings, on="book_id")#.astype({'rating':'float32'})
userpiv=bookratings.pivot_table(index="user_id",columns="book_id",values="rating")
#print(userpiv.head())
userpiv=userpiv.sub(userpiv.mean(axis=1),axis=0).fillna(1)
npuserpiv=userpiv.values
print(userpiv)
print("yeet")
#np.linalg.svd(userpiv)
u, sigma, vt = svds(userpiv,k=50)
sigma = np.diag(sigma)

vt = sigma @ vt
v=vt.T
print(v.shape)
film = v[15]
vdistto = np.sum(v-film, axis = 1).reshape(-1,1)
print(vdistto)
print(np.asarray([userpiv.index]).T.shape)
print(np.concatenate([np.asarray([userpiv.index]).T,u],axis = 1))
print(np.concatenate([np.asarray([userpiv.columns]).T,vdistto],axis = 1))
#print(p[p[:,1].argsort()])
