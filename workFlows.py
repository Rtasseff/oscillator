#!/usr/bin/env python
#
# 
#     Copyright (C) 2003-2012 Institute for Systems Biology
#                             Seattle, Washington, USA.
# 
#     This library is free software; you can redistribute it and/or
#     modify it under the terms of the GNU Lesser General Public
#     License as published by the Free Software Foundation; either
#     version 2.1 of the License, or (at your option) any later version.
# 
#     This library is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#     Lesser General Public License for more details.
# 
#     You should have received a copy of the GNU Lesser General Public
#     License along with this library; if not, write to the Free Software
#     Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307  USA
# 
"""
Python module to run oscilator work flows.
Takes a time seris, finds relevent frequanies,
calculates relevent phase varriables, identifies
coupling coefficents for simple coupled oscillator
system.

time course:
data provided in 2d np matrix, col=variables and row=observations
time provided in 1d np array, values corrispond to data rows

created 20120727 RAT
"""

import numpy as np
from oscillator import variables
from oscillator import estFreq
from SLR2 import SLR2_2
import sys
from genTools import manageData



def simplest(dataPath,timePath,outPath):
	""" Runs a simple work flow:
	w determined by defult code,
	estimate freqs by orthogonal 
	multi method, parameters 
	hardcoded and chosen for quicker less 
	accurate results.
	"""
	# get the data
	data = np.loadtxt(dataPath)
	t = np.loadtxt(timePath)
	n,m = data.shape
	# get the proper frequancies
	Y,w = estFreq.est1(data,t,nPerm=100)
	# get the phase variables
	th,dth = makeVarMat(data,t,w,Y)
	# just getting coef, nothing else!
	K = np.zeros((m,m+1))
	# do each fit
	for i in range(m):
		enm = runFit(th,dth,i)
		K[i,enm.indices] = enm.coef
		K[i,-1] = enm.intercept[0]

	np.savetxt(outPath,K)
	

def makeVarMat(data,t,w,Y):
	n,m = data.shape
	th = np.zeros((n,m))
	dth = np.zeros((n,m))
	for i in range(m):
		xHat,p,f,a = variables.getVars(t,x=data[:,i],w=w[Y[:,i]])
		th[:,i] = p
		dth[:,i] = f
	return(th,dth)
	
def runFitnEst(th,dth,i):
	""" runs a defult SLR fit using 
	a lasso, also estimates std err of 
	coef and error of model (ie full solution).
	"""
	y = dth[:,i]
	X = np.sin(th.T-th[:,i]).T
	solution,enm = SLR2_2.estModel(X,y)
	return(solution,enm)
	
def runFit(th,dth,i):
	""" runs a defult SLR fit using 
	a lasso with bs to estimate penalty.
	"""
	y = dth[:,i]
	X = np.sin(th.T-th[:,i]).T
	enm = SLR2_2.select(X,y)
	return(enm)


def main():
	method = sys.argv[1]
	dataPath = sys.argv[2]
	timePath = sys.argv[3]
	outPath = sys.argv[4]
	
	if method=='simplest':
		simplest(dataPath,timePath,outPath)
	else:
		print 'wrong method selected'

if __name__ == '__main__':
	main()
