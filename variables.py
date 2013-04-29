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
Python module that calculates various varriables that represent a
a periodic signal.  Assumes the signal is represented as fourier seris (FSM):
each row is associated with a discreet frequency
col 0 - frequency 
col 1 - constant 
col 2 - cos constant
col 3 - sin constant

NOTE: in this code we ignore the constants (ie they are subtracted from 
the signal, we are only considering the periodic component).

created rat 20120725
edited: 
rat 20120807 added discreat FFT method of signal analysis
RAT 20120808 added a methods to create the FSM 
	note this can be used to by pass the need for w
	typically we expect the user to preselect a w
	in the abscence of w we assume that its not periodic
	if the user calls the genFSM first the uses that output
	for input into getVars a defult w will always be selected
RAT 20120920 added method to randomly shift the data
	we introduce a random phase shift for each frequancy 
	This can be done to compare our signal results to 
	a random one that has similar properties (same
	frequancy distribution) 
	
RAT 20121207 added the correction for dampening 
	this is in concert with a similar addition
	in the fourierSeris.  In breif it adds
	an a priori dampening to the regression for 
	data that may have a damped osilitory signal 
	and we still want to recover the underlying 
	oscilitory signal.  Here we will add that feature
	and consider it as an outside force on the underlying 
	oscilaiton.  Like in the loss of pop sync for cell cycl 
	exp.  It basiclly changes the amplitude, which we
	typicaly dont care about anyways.
 
	

"""


import numpy as np
from robustFourierSeries import fourierSeries
from genTools import dataUtil
from scipy import signal



def getVars(time,x=[],xFSM=[],w=[],randomShift=False,damp=0):
	"""get the instintanious phase and frequancy of the discrete
	time course signal x at times time.
	If w is provided then xFSM (the fouirer rep of x) is 
	cacluated at the provided ws and x must be provided.
	If w is not provided then xFSM is assumed to be 
	provided and will be used for the calculations.
	if xFSM is provided and x is not then
	x is estimated at the appropriate times
	using xFSM.
	The randomShift flag was added to construct 
	randomized varriables, if true we randomly 
	shift each of the frequancy terms contributing 
	to the signal and return varriables for that.
	returns
	all corrispond to the points in time
	all are numpy 1d arrays
	x	real signal, after preprocessing
	p	instintanious phase
	f	instintanious frequancy 
	amp	instintanious amplitude	
	"""
	
	if len(w)>0:
		# if given w then by convention always use it to estimate an FSM 
		xFSM = getFSM(x,time,w,damp)

	# check if not periodic
	# the way I use this latter I run many signals through 
	# with out checking if a w is associated 
	# its easier for me to just return a set of zeros
	# for safty I have included a message to let other user know this 
	if len(xFSM)==0 and len(w)==0:
		xFSM = np.zeros(4)
		print 'no frequancy or FSM info, assuming no periodicity'
		if len(x)==0:
			x = np.zeros(len(time))


	# here we should have an FSM no matter what
	# we may have x's or not, if yes lets correct them where needed
	if len(x)>0:
		
		# in this code we remove the constants and focus on periodic component
		# we could have a matrix or an array depending on number of frequancies
		if len(xFSM.shape)==2:
			n,m = xFSM.shape
			x = x-np.sum(xFSM[:,1])
			xFSM[:,1] = np.zeros(n)
		elif len(xFSM.shape)==1 :
			x = x-xFSM[1]
			xFSM[1] = 0
		else:
			raise ValueError('xFSM has wrong dimension?')

		# we also have to correct for the dampening if it exists to focus on periodic component
		if np.abs(damp)>1E-22:
			x = x/np.exp(-damp*time)



	# if we do not have the x's do what we need to get all needed values
	if len(xFSM)>0 and len(x)==0:
		# we have an FSM but no x, lets estimate it

		# Again in this code we remove the constants and focus on periodic component
		# we could have a matrix or an array depending on number of frequancies
		if len(xFSM.shape)==2:
			n,m = xFSM.shape
			xFSM[:,1] = np.zeros(n)
		elif len(xFSM.shape)==1 :
			xFSM[1] = 0
		else:
			raise ValueError('xFSM has wrong dimension?')
		

		x =  FSM2values(xFSM,time)

	
	# by this poin the xFSM only has the peridoic comp and x has been corrected as well
	# its possible we were given an x with NAN values, correction leaves them nan, 
	# use fit to replace those values 
	if np.any(np.isnan(x)):
		x = replaceNAN(x,time,xFSM)

	
	
	# if the randomShift flag, lets prep for that 
	if randomShift:
		xFSM = getRandomPhaseShiftFSM(xFSM)
		x = FSM2values(xFSM,time)

	# ok now lets get all the values get the hilbert transform 
	xHilFSM = hilbert(xFSM)
	# get the values of xHil at the correct times
	xHil =  FSM2values(xHilFSM,time)
	# get the instintanious phase varriables 
	p = phaseInstant(x,xHil)
	# get the instintanious frequancies  
	f = frequancyInstant(x,xFSM,xHil,xHilFSM,time)
	# for shits get instantanious amp 
	amp = amplitudeInstant(x,xHil)

	return(x,p,f,amp)
	
def hilbert(x):
	"""given the FSM for the signal, x, generate the
	FSM that defines the hilbert transform of x
	"""
	# we could have a matrix or an array depending on number of frequancies
	if len(x.shape)==2:
		n,m = x.shape
		# Ignores the constants as they do not make sense here
		xHil = np.zeros((n,m))
		xHil[:,0] = x[:,0] 
		# all cos go to sin and all sin go to negative cos
		xHil[:,3] = x[:,2]
		xHil[:,2] = -1*x[:,3]
	elif len(x.shape)==1:
		# Ignores the constants as they do not make sense here
		xHil = np.zeros(4)
		xHil[0] = x[0] 
		# all cos go to sin and all sin go to negative cos
		xHil[3] = x[2]
		xHil[2] = -1*x[3]
	else:
		raise ValueError('x has wrong dimension?')

	
	return(xHil)




def FSM2values(FSM,time,damp=0):
	""" find the values of the signal described by
	FSM at the time points in time.
	"""
	
	
#	# we could have a matrix or an array depending on number of frequancies
#	if len(FSM.shape)==2:
#		xHat = np.zeros(len(time))
#		n,m = FSM.shape
#		# looping through w is probably faster then looping through t 
#		# python does not do well with loops and w typically smaller the t
#		for i in range(n):
#			w = FSM[i,0]
#			xHat = xHat + FSM[i,1] + FSM[i,2]*np.cos(time*w)+ FSM[i,3]*np.sin(time*w)
#	elif len(FSM.shape)==1:
#		xHat = FSM[1] + FSM[2]*np.cos(time*FSM[0])+ FSM[3]*np.sin(time*FSM[0])
#	else:
#		raise ValueError('x has wrong dimension?')

	# going to use the built in method in fourierSeris to do this
	if FSM.ndim == 1 :
		xHat = fourierSeries.estSignal(FSM[1:],FSM[0],time,damp)
	elif FSM.ndim==2 :
		xHat = fourierSeries.estSignal(FSM[:,1:],FSM[:,0],time,damp)
	else:
		raise ValueError('x has wrong dimension?')

	return(xHat)
		
		


def phaseInstant(x,xHil):
	"""given discrete signals for both x and xHil (the hilbert
	transfor of x) we clculate the signed angel in radians 
	between them.
	"""
	return(np.arctan2(xHil,x))

def timeDif(x):
	"""given the FSM for a signal calculates the 
	FSM representation of the derivative wrt time
	"""

	# we could have a matrix or an array depending on number of frequancies
	if len(x.shape)==2:
		n,m = x.shape
		xDot = np.zeros((n,m))
		w = x[:,0]
		xDot[:,0] = w
		# get the current cos constants
		a1 = x[:,2]
		# get the current sin constants
		a2 = x[:,3]
		# the new cos come from sin
		xDot[:,2] = w*a2
		# the new sin coef are negatives of cos
		xDot[:,3] = -1*w*a1
	elif len(x.shape)==1:
		xDot = np.zeros(4)
		w = x[0]
		xDot[0] = w
		# get the current cos constants
		a1 = x[2]
		# get the current sin constants
		a2 = x[3]
		# the new cos come from sin
		xDot[2] = w*a2
		# the new sin coef are negatives of cos
		xDot[3] = -1*w*a1
	else:
		raise ValueError('x has wrong dimension?')


	return(xDot)

def frequancyInstant(x,xFSM,xHil,xHilFSM,time):
	"""Given the discrete signal x, the FSM 
	rep, xFSM the discrete hill transform of x,
	xHil and the FSM rep of the hil transform, 
	xHil, we calculate the freq
	which is the time derivative of the phase.
	We return a discrete derivative evaluated 
	at the points in time, corrisponding to x.
	"""
	left = x*FSM2values(timeDif(xHilFSM),time)
	right = xHil*FSM2values(timeDif(xFSM),time)
	bottom = x**2+xHil**2
	# this has sort of removed a 0, divided with 
	# assumption not zero for simplification
	# so this solution works if the dpdt is not 
	# already zero, if it is then we have a problem 
	# lets make sure that none are zero
	tmp = bottom==0
	# is there are zeros we can set any constant 
	# as numerator has to be zero
	bottom[tmp] = 1
	return((left-right)/bottom)

def amplitudeInstant(x,xHil):
	""" given the discrete data x and the
	hilbert transform of x, xHil, calculate the
	instintanious amplitude.
	"""
	return(np.sqrt(x**2+xHil**2))

def replaceNAN(x,time,xFSM):
	"""given a discrete data set, x, 
	corrisponding to the rime points in time and a
	FSM reprisintation of x, xFSM, estimate 
	and replace any nan values and return 
	a new discrete data set.
	"""
	z = np.isnan(x)
	xEst = FSM2values(xFSM,time)
	x[z] = xEst[z]
	return(x)

def getVarsDFT(x,t):
	"""Uses the discrete fast fouier transform to calculate
	the analytic signal (which is the hilbert transform 
	of the FS) this is diffrent as it uses the numpy solvers 
	to calculate the DFT, which is not robust and designed 
	for uniform sampling, and does not account for the dampening.
	Then the instantnious phase is calculated (and amp).
	The instantanious Freq is estimated by finite difference.
	x	1d np array of data
	t	1d np array of time corrisponding to x
		t will be used to sort 
		the x value, redundant data at a time point 
		will be combine via median.
	returns
	x	new x value (if redundent or unordered it will be diffrent)
	t	new corrisponding time
	th	instantanious phase
	dth	instantanious freq
	r	inst... amp...
	"""
	t,x = dataUtil.aveUnqTime(t,x,method='median')
	z = signal.hilbert(x)
	th = np.unwrap(np.arctan2(z.imag,z.real))
	r = np.abs(z)
	dth,t = dataUtil.FD(th,t)
	return(x,t,th,dth,r)

def getFSM(x,time,w=[],damp=0):
	"""each row is associated with a discreet frequency
	col 0 - frequency 
	col 1 - constant 
	col 2 - cos constant
	col 3 - sin constant
	"""
	# we can handel nan values in the fit, just one less value
	# lets use a tool from the fit module to remove any nan values
	x,time = fourierSeries.removenan(x,time)
	if len(w)==0:
		w = fourierSeries.defultFreq(time)

	B,b = fourierSeries.calcPeriodogram2(x,time,w,damp) # note, this can handel nan values
	# we could have a matrix or an array depending on number of frequancies
	if len(b.shape)==2:
		xFSM = np.c_[w,b]
	else:
		xFSM = np.append(w,b)

	return(xFSM)

def getRandomPhaseShiftFSM(xFSM):
	"""Creates an FSM that represents
	an added random phase shift (uniform
	from 0 to 2pi) to the trajectories 
	represented by xFSM.
	"""
	# signal a*sin(wt) + b*cos(wt)
	# can be represented as 
	# Aprime*sin(wt+theta)
	if xFSM.ndim ==2:
		b = xFSM[:,2]
		a = xFSM[:,3]
		n = len(xFSM)
		theta = np.zeros(len(b))
		theta[np.abs(b)>1E-21] = np.arctan(b/a)
	else:
		b = xFSM[2]
		a = xFSM[3]
		n = 1
		if b<1E-21: theta = 0
		else: theta = np.arctan(b/a)
	Aprime = a/np.cos(theta)
	# we can remove the theat and
	# add a rand shift to the signal
	# Aprime*sin(wt+0+shift)
	# equivlant to starting at random phases
	# then solve for new coef
	# Ahat*sin(wt) + Bhat*cos(wt)
	shift = np.random.rand(n)*np.pi*2
	Ahat = Aprime*np.cos(shift)
	Bhat = Aprime*np.sin(shift)
	if xFSM.ndim==2:
		shiftFSM = np.zeros((n,4))
		shiftFSM[:,0] = xFSM[:,0]
		shiftFSM[:,1] = xFSM[:,1]
		shiftFSM[:,2] = Bhat
		shiftFSM[:,3] = Ahat
	else:
		shiftFSM = np.zeros(4)
		shiftFSM[0] = xFSM[0]
		shiftFSM[1] = xFSM[1]
		shiftFSM[2] = Bhat
		shiftFSM[3] = Ahat
	
	return shiftFSM
