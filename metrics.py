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
Functions and scripts for calculating and analyzing various
order parameters or other metrics for oscillitory systems.
created: RAT 20120807
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec

def orderParam(th,q=1):
	"""Calculate the complex order parameter:
	z = r*exp(phi*i)= 1/N \sum(exp(q*th_m*i))
	where q == 1 is the traditional order parameter 
	for phase coupled oscillators.
	th	2d array (or 1d) with the
		col as oscillators and the row
		as observations 
	q	int, order of harmonic
	return 
	z
	"""
	tmp = np.exp(q*th*1j)
	if tmp.ndim==2:
		z = np.mean(tmp,1)
	elif tmp.ndim==1:
		z = np.mean(tmp)
	else:
		raise ValueError('bad ndim on th')
	return(z)



def plot(th,t=[],title='*',fname='*',nBins=1000):
	z1 = orderParam(th,1)
	z2 = orderParam(th,2)
	gs = gridspec.GridSpec(2, 2)
	ax1 = plt.subplot(gs[0,:])
	ax2 = plt.subplot(gs[1,0],polar=True)
	ax3 = plt.subplot(gs[1,1])

	if len(t)>0:
		ax1.plot(t,np.abs(z1),'r-',label='r1')
		ax1.plot(t,np.abs(z2),'b-',label='r2')
	else:
		ax1.plot(np.abs(z1),'r-',label='r1')
		ax1.plot(np.abs(z2),'b-',label='r2')
	ax1.legend()
	ax1.set_xlabel('time')
	ax1.set_ylabel('r')
	ax1.set_ylim([0,1.1])
	if title!='*':
		ax1.set_title(title)
	
	
	ax2.plot(np.arctan2(z1.imag,z1.real),np.abs(z1),'r',label='z1')
	ax2.plot(np.arctan2(z2.imag,z2.real),np.abs(z2),'b',label='z2')
	ax2.set_ylim([0,1.1])
	ax2.legend()

	if th.ndim==2:
		thLast = wrap(th[-1,:])
	else:
		thLast = wrap(th)
	ax3.hist(thLast,nBins,label='th(t=-1)')
	ax3.set_xlim([-np.pi,np.pi])
	ax3.legend()

	#ax3.plot(np.cos(thLast),np.sin(thLast),'o',label='th(t=-1)')
	#ax3.legend()

	if fname!='*':
		plt.savefig(fname)
		plt.clf()
	else:
		plt.show()



def pairedSycnStr(th):
	""" using the th values (instantanious phase)
	we calculate the paired synchronization 
	strengths (averaged over observations) for 
	each oscillator pair
	th 	matrix rows are observations, 
		cols are oscillators
	This quantity is defined in Allefeld2002
	"""
	n,m = th.shape
	R = np.ones((m,m))
	for i in range(m):
		for j in range(i):
			delta = th[:,i] - th[:,j]
			tmp = orderParam(delta)
			R[i,j] = np.abs(tmp)
			R[j,i] = np.abs(tmp)

	return R

def clusterSyncStr(th=[],R=[]):
	""" Using the paired synchronization
	strengths R, find the cluster sysnchronization 
	strength for each oscillator
	R is from pairedSycnStr
	This quantity is defined in Allefeld2004
	and the method is defined in Kim2008
	th is the pahse matrix row obs col var
	if th is provided R is not requiered and
	will be calculated using th.
	"""
	if len(th)>0:
		R = pairedSycnStr(th)

	p = np.mean(R,1)
	n = len(p)
	eps = 1E-26
	maxIter = 100
	tol = 1E-5
	tolMet = False
	for j in range(maxIter):
		pOld = p
		for k in range(n):
			num = 0
			den = 0
			for i in range(n):
				if i!=k:
					tmp = (1-(p[i]*R[i,k])**2)**2
					if tmp > eps:
						F = 1./tmp
					else:
						F = 1./eps
					num = num + F*p[i]*R[i,k]
					den = den + F*(p[i]**2)
			p[k] = .5*(p[k] + num/den)
		
		if np.mean((p-pOld)**2)<tol:
			tolMet = True
			break
	if tolMet:
		print 'tolerance met :)'
	else:
		print 'maxIter met :('
	
	
	return(p)	

def syncAnalysis(X,t,w=[]):
	"""Perform the synchronization analysis
	as in Allefeld2004
	Some issues were identifed here, the code is 
	right (I think) but the algoritham does
	not do what would be expected.
	"""
	import variables
	n,m = X.shape
	# get all the FSM
	print 'estimating DFS for all '+str(m)+' oscillators'
	xFSM = []
	for i in range(m):
		if len(w)>0:
			xFSM.append(variables.getFSM(X[:,i],t,w))
		else:
			# estimate w if here (should only happen first time
			xFSM.append(variables.getFSM(X[:,i],t))
			# use estimated w
			w = xFSM[i][:,0]
	
	# now calculate the the synchronization at each w
	k = len(w)
	R = np.zeros((k,m))
	tHat = np.arange(np.min(t),np.max(t),(np.max(t)-np.min(t))/float(len(t)))
	nTime = len(tHat)

	for i in range(k):
		print 'calculating sync at freq '+str(w[i])
		th = np.zeros((nTime,m))
		for j in range(m):
			x,phase,f,amp = variables.getVars(tHat,xFSM=xFSM[j][i,:])
			th[:,j] = phase 


		R[i,:] = clusterSyncStr(th)
		
	R = np.c_[w,R]
	return(R)

def freqTimeOrder(X,t,w=[],q=1,nTime=0):
	"""Perform the synchronization analysis
	similar to Allefeld2004, still not quite 
	what I want.
	"""
	import variables
	n,m = X.shape
	# get all the FSM
	print 'estimating DFS for all '+str(m)+' oscillators'
	xFSM = []
	for i in range(m):
		if len(w)>0:
			xFSM.append(variables.getFSM(X[:,i],t,w))
		else:
			# estimate w if here (should only happen first time
			xFSM.append(variables.getFSM(X[:,i],t))
			# use estimated w
			w = xFSM[i][:,0]
	
	# now calculate the the synchronization at each w
	k = len(w)
	if nTime==0: nTime = len(t)
	tHat = np.arange(np.min(t),np.max(t),(np.max(t)-np.min(t))/float(nTime))
	nTime = len(tHat)
	R = np.zeros((k,nTime))

	for i in range(k):
		print 'calculating sync at freq '+str(w[i])
		th = np.zeros((nTime,m))
		for j in range(m):
			x,phase,f,amp = variables.getVars(tHat,xFSM=xFSM[j][i,:])
			th[:,j] = phase 


		R[i,:] = np.abs(orderParam(th,q))
		
	return(R,tHat,w)

def wrap(th):
	"""Wraps th to the interval -pi to pi"""
	if min(th)<-1*np.pi-1E-21 or max(th)>np.pi+1E-21:
		thNew = th.copy()
		if thNew.ndim==2:
			n,m = thNew.shape
			for i in range(n):
				for j in range(m):
					while thNew[i,j]<-1*np.pi-1E-21:
						thNew[i,j] = thNew[i,j]+2*np.pi
					while thNew[i,j]>np.pi+1E-21:
						thNew[i,j] = thNew[i,j]-2*np.pi

		elif thNew.ndim==1:
			n = len(thNew)
			for i in range(n):
				while thNew[i]<-1*np.pi-1E-21:
					thNew[i] = thNew[i]+2*np.pi
				while thNew[i]>np.pi+1E-21:
					thNew[i] = thNew[i]-2*np.pi
		return(thNew)
	else:
		return(th)

