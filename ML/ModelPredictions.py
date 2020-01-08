import pandas as pd
import numpy as np
import pickle
def mlpredict(features):
	result=[]
	with open("decisionboundary.pckl", "rb") as f:
	    while True:
	        try:
	            clf=pickle.load(f)
	        except EOFError:
	            break
	pred = clf.predict([features])
	result.append(pred[0])
	
	with open("decisiontree.pckl", "rb") as f:
	    while True:
	        try:
	            clf=pickle.load(f)
	        except EOFError:
	            break
	pred = clf.predict([features])
	result.append(pred[0])
	with open("ensemblelearning.pckl", "rb") as f:
	    while True:
	        try:
	            clf=pickle.load(f)
	        except EOFError:
	            break
	pred = clf.predict([features])
	result.append(pred[0])

	with open("instancebased.pckl", "rb") as f:
	    while True:
	        try:
	            clf=pickle.load(f)
	        except EOFError:
	            break
	pred = clf.predict([features])
	result.append(pred[0])

	print result