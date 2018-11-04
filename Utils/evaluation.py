import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from geopy.distance import vincenty
import pandas as pd
import csv
#import seaborn as sns


def csv2dict(path):
	reader = csv.DictReader(open(path))
	my_dict = {rows['messageid']: np.array((rows['latitude'],rows['longitude'])) for rows in reader}
	return my_dict

def df2dict(df):
	return {row['messageid']: [row['latitude'], row['longitude']] for _, row in df.iterrows()}

def vincenty_vec(coord1, coord2):
	"""
	Returns the pointwise Vincenty distance between two GPS-coords arrays

	Parameters
	----------
	coord1 : dict {'messageid': [latitude', 'longitude']}
	coord2 : dict {'messageid': [latitude', 'longitude']}

	Returns
	-------
	m : ndarray 
	    m[i] = vincenty_distance(coord1[i], coord2[i])

	"""
	assert(set(coord1.keys())==set(coord2.keys()))
	vin_vec_dist = [(vincenty(coord1[k], coord2[k])).km for k in coord1]
	return vin_vec_dist


def criterion(y_pred, y_true):
	error_vector = vincenty_vec(y_pred, y_true)
	return np.percentile(error_vector, 90)


def plot_error(y_pred, y_true):
	error_vector = vincenty_vec(y_pred, y_true)
	
	f = plt.figure()
	ax = f.add_subplot(111)
	plt.hist(error_vector, cumulative=True, histtype='step', density=True, bins=500)

	plt.vlines(x=np.percentile(error_vector, 90), ymin=0, ymax=1, colors='r', label='criterion')
	plt.xlabel('Distance Error (km)')
	plt.ylabel('Cum Proba (%)')
	plt.xlim(-.1, 100)
	f.legend()
	f.show()







