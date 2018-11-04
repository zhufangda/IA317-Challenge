import sys
import os 
import os.path as op
import json
import logging
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from collections import deque
from scipy import sparse
import pandas as pd 
import numpy as np
import scipy 
from scipy import sparse
#from tqdm import tqdm
import geopy.distance as distance
from scipy.spatial import KDTree
import functools
#import joblib
import copy

sys.setrecursionlimit(10000)

#foi_folder = sys.argv[1]
#data_folder = sys.argv[2]

dict_of_gby = {'rssi': ['bsid'],
				'freq': ['bsid'],
				'latitude_bs': ['bsid'],
				'longitude_bs': ['bsid'],
				'latitude': [''],
				'longitude': [''],
				'speed': [''],
				'dtid': [''], 
				'did': [''],}

#features_of_interest = json.loads(open(foi_folder, 'r').read())

def _build_features_dict(data_frame, features_of_interest):

	if 'latitude_bs' in list(features_of_interest.values())[0] or 'latitude_bs' in list(features_of_interest.values())[1]:
		sys.stdout.write(u"\u001b[4mDownloading base stations informations (take a coffee, can take a while)\u001b[0m \u001b[1m\u001b[0m \n")
		from sigfox.datamart_data import BasestationData
		basestationdata = BasestationData('/Users/kevinelgui/Thèse/Projet/Data/GeoDataFrame', **{'basestation_ids': bsid_unique}).dataframe
		basestationdata = basestationdata[basestationdata.objid.isin(bsid_unique)]
		data_frame['latitude_bs'] = data_frame.apply(lambda row: bs_dict[row['bsid']]['latitude'], 1)
		data_frame['longitude_bs'] = data_frame.apply(lambda row: bs_dict[row['bsid']]['longitude'], 1)
	#Verbose
	bsid_nunique = data_frame.bsid.nunique()
	bsid_unique =  data_frame.bsid.unique()
	did_nunique = data_frame.did.nunique()
	sys.stdout.write(u"\u001b[4mFeatures of interest:\u001b[0m \u001b[1m {} \u001b[0m \n".format(
	features_of_interest.get('features_of_interest')))
	sys.stdout.write(u"\u001b[4mTarget:\u001b[0m \u001b[1m {} \u001b[0m \n".format(
	features_of_interest.get('target')))
	sys.stdout.write(u"\u001b[4mNb of base stations:\u001b[0m \u001b[1m {} \u001b[0m \n".format(bsid_nunique))
	sys.stdout.write(u"\u001b[4mNb of unique DeviceId train:\u001b[0m \u001b[1m {} \u001b[0m\n".format(str(did_nunique)))

	l0 = len(data_frame)

	## Discretization
	
	#sys.stdout.write(u"\u001b[4mDiscretization\u001b[0m:~ 0.300 m x 0.300 m square\n")

	# LON_grid, LAT_grid = np.meshgrid(np.arange(data_frame.longitude.min()-1e-3,
	#                                           data_frame.longitude.max()+1e-3, 1e-2),
	#                                 np.arange(data_frame.latitude.min()-1e-3,
	#                                           data_frame.latitude.max()+1e-3, 1e-2))

	# Tree = KDTree(np.column_stack((LAT_grid.ravel(), LON_grid.ravel())))
	# classes_ = {i: tuple(p) for i, p in enumerate(Tree.data)}
	# classes = Tree.query(data_frame[['latitude', 'longitude']].values)[1]
	# data_copy = data_frame.copy()
	# data_copy['label'] = classes

##

	st = {f.__add__(x) for f in features_of_interest['features_of_interest'] for x in dict_of_gby[f]}
	feature_name = set({})
	for f in features_of_interest['features_of_interest']:
		for x in dict_of_gby[f]:
			if x:
				g = data_frame.groupby(x).groups
				for gg in g:
					feature_name = feature_name.union({f.__add__(str(gg))})
			else:
				feature_name = feature_name.union({str(f)})
	
	target_name = set({})
	for f in features_of_interest['target']:
		for x in dict_of_gby[f]:
			if x:
				g = data_frame.groupby(x).groups
				for gg in g:
					target_name = target_name.union({f.__add__(str(gg))})
			else:
				target_name = target_name.union({str(f)})
	# feature_name = {'RSSbs{bsid}'.format(bsid=x) for x in 
    #                bsid_unique}
	# feature_name = feature_name.union({'FreqRec{bsid}'.format(bsid=x) for x in 
	#                 bsid_unique})
	# #feature_name = feature_name.union({'LatLonGrid{bsid}'.format(bsid=x) for x in 
	# #                bsid_unique})
	# feature_name = feature_name.union({'Speed'})
	# #feature_name = feature_name.union({'Period_since_{n}past_msgs_'.format(n=i) for i in np.arange(6)})
	# feature_name = feature_name.union({'LatLonHistory_{n}'.format(n=i+1) for i in np.arange(0, 10)})
	# #feature_name = feature_name.union({'Type_id_{}'.format(dtid) for dtid in df.dtid.unique()})
	# #feature_name = feature_name.union({'Device_id_{}'.format(did) for did in did_unique})
	feature_dict = {name: i for i, name in enumerate(feature_name)}
	target_dict = {name: i for i, name in enumerate(target_name)}
	return (feature_dict, feature_name), (target_dict, target_name)

def my_parser(data, label, feature_dict, target_dict, features_of_interest):
	
	data_copy = pd.concat((data, label), 1)
	
	res = {'dict_X': feature_dict, 'dict_y': target_dict} 

	data_it = copy.deepcopy(data_copy)
	
	groupby_msgid = data_it.groupby(['messageid', 'time_ux'],
						sort=True, group_keys='time_ux')
	##First we derive the vector of label y
	#y = groupby_msgid[features_of_interest['target']].first()
	
	#Last_time_ = data_copy.groupby('did', sort=True, group_keys='time_ux')[['label', 'time_ux']]
	
	features_, I, J  = [], [], []
	
	y, I_output, J_output = [], [], []
	msg_list = []
	it = enumerate(groupby_msgid).__iter__()

	for ind, ((msg, time_ux), value) in it:
		msg_list.append(msg)
		time_ux = time_ux//1000
		did = value.did.unique()[0]
		#histo = Last_time_.get_group(did)
		#histo_latence = histo[(histo.time_ux//1000 - time_ux)>3600]
		#histo_latence_classe = histo_latence.label.sort_values(ascending=False)

		# for n, c in enumerate(histo_latence_classe.values):
		# 		if n > 9: 
		# 			break
		# 		else:
		# 			I.append(ind)
		# 			J.append(feature_dict['LatLonHistory_{}'.format(n+1)])
		# 			data.append(c)

# 	    dtid = value.dtid.unique()[0]
#	    classe = value.classe.unique()[0]
# 	    History[did].appendleft(classe)
# 	    Time[did].appendleft(time_ux)
# 	    delay = Time[did][0] - Time[did][-1]
# 	    Period[did].appendleft(0)
# 	    new_period = deque(**{'maxlen':6})

# 	    #for d in Period[did]:
# 	    #    new_period.appendleft(d + delay)

# 	    #Period[did] = new_period

# 	    design_dictionary = defaultdict(defaultdict)

# 	    #I.append(ind)
# 	    #J.append(feature_dict['Type_id_{}'.format(dtid)])
# 	    #data.append(1)
# 	    #I.append(ind)
# 	    #J.append(feature_dict['Device_id_{}'.format(did)])
# 	    #data.append(1)
# 	    #I.append(ind)
# 	    #J.append(feature_dict['Speed'])
# 	    #data.append(np.nan_to_num(value.speed.values[0]))

# 	    #for n, tt in enumerate(Period[did]):
# 	    #    I.append(ind)
# 	    #    J.append(feature_dict['Period_since_{}past_msgs_'.format(n)])
# 	    #    data.append(tt)

# 	    #for n, tt in enumerate(History[did]):
# 	    #    if not n==0:
# 	    #        I.append(ind)
# 	    #        J.append(feature_dict['LatLonHistory_{}'.format(n)])
# 	    #        data.append(tt)
		
		"""
		fill y 
		"""
		for k in features_of_interest['target']:
			for gb_key in dict_of_gby[k]:
				if gb_key:
					for g, val_gb in value.groupby(gb_key):
							I_output.append(ind)
							J_output.append(target_dict[k.__add__(str(g))])
							y.append(val_gb[k].values[0])
				else:
					I_output.append(ind)
					J_output.append(target_dict[k])
					y.append(value[k].values[0]) 
	# 	        I.append(ind)
	# 	        J.append(feature_dict['LatLonGrid{}'.format(bsid)])
	# 	        data.append(bs_classe[bsid])
		"""
		fill data 
		"""
		for k in features_of_interest['features_of_interest']:
			for gb_key in dict_of_gby[k]:
				if gb_key:
					for g, val_gb in value.groupby(gb_key):
							I.append(ind)
							J.append(feature_dict[k.__add__(str(g))])
							features_.append(1*(val_gb[k].max()< -1))
				else:
					I.append(ind)
					J.append(feature_dict[k])
					features_.append(1*(value[k].max()<-1))

	coo_matrix_ = sparse.coo_matrix((features_, (I, J)), shape=(len(groupby_msgid), len(feature_dict)), dtype=np.int64)
	coo_matrix_y = sparse.coo_matrix((np.array(y)[:], (I_output, J_output)), shape=(len(groupby_msgid), len(target_dict)), dtype=np.float32)
	res.update({'X': coo_matrix_, 'y': coo_matrix_y, 'msg_list': msg_list })

	return res

#my_parser = memory_parseddata.cache(_my_parser)

#if __name__== '__main__':
#	(feature_dict, feature_name), (target_dict, target_name)  = _build_features_dict(data_folder, features_of_interest)
#	my_parser(data_folder, feature_dict, target_dict)

