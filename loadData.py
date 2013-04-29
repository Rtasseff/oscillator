"""
Python2.7 script to load data in this folder related to cell cycle coupled oscillators
20130201 RAT
"""

from genTools import manageData
import numpy as np

def loadStudy(name):

	if name == 'GSE3635':
		dataDir = '/Users/rtasseff/Projects/CoupledOscillator/cellCycle/GSE3635/rawData'
		meta, data, sampleData, featureData = manageData.load_exp(dataDir)
		t = sampleData['time']
		x = data
		# temp hack
		if t[0] < 1E-20:
			t = t[1:]
			x = x[1:]
		
		infoDir = '/Users/rtasseff/Projects/CoupledOscillator/cellCycle/GSE3635/periodicity/ID_20130131'
		# useable for this guy frequancies 
		w = np.loadtxt(infoDir+'/w.dat')
		# indicies for this guy
		# currently these are p<.05 linear corrected 
		index = np.array(np.loadtxt(infoDir+'/linearCorrection/index.dat'),dtype=int)
		
		study = {}
		study['X'] = x
		study['t'] = t
		study['w'] = w
		study['index'] = index
		study['meta'] = meta
		
	elif name == 'GSE5283':
		dataDir = '/Users/rtasseff/Projects/CoupledOscillator/cellCycle/GSE5283/rawData'
		meta, data, sampleData, featureData = manageData.load_exp(dataDir)
		t = sampleData['time']
		x = data
		# temp hack
		if t[0] < 1E-20:
			t = t[1:]
			x = x[1:]
		
		infoDir = '/Users/rtasseff/Projects/CoupledOscillator/cellCycle/GSE5283/periodicity/ID_20130131'
		# useable for this guy frequancies 
		w = np.loadtxt(infoDir+'/w.dat')
		# indicies for this guy
		# currently these are p<.05 linear corrected 
		index = np.array(np.loadtxt(infoDir+'/linearCorrection/index.dat'),dtype=int)
		
		study = {}
		study['X'] = x
		study['t'] = t
		study['w'] = w
		study['index'] = index
		study['meta'] = meta
	else:
		raise ValueError(' cannot find name: '+name)	

		
	return(study)
