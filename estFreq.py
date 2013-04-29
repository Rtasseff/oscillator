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
Python module to estimate the frequancies of multiple signals.
Signals are treated independently.
Frequancies are determined using a robust regression method,
statistically significant frequancies are returned.

created 20120727 RAT
"""
import numpy as np
from robustFourierSeries import fourierSeries



def est1(x,t,w=[],nPerm=1000,alpha=.05):
	"""Estimates the frequancies that are significant.
	Use a method that calculated p-values for each 
	frequancy.  However, the fit is done using all
	frequancies in an iterative fashion with the removal 
	of the previous fit.  
	x	data, 2d numpy matrix, col are the variable
		the rows are the obsevations
	t	time corrisponding to x rows, 1d np array
	w	freqs to consider, if not set
		will be chosen with small steps.	
		1d np array
	nPerm	number of permutations in the test
	alpha	the statistical significance cut off
	returns	
	Y	chosen freq, 2d np boolean matrix 
		each col corrispond to x variables
		the rows corrispond to freq in w. 
	w	the freq used (if provided will be the same)




	NOTE:
	* The down side to this method is that
	a poor choice of freq range can result in 
	large coef for extranous freq due to noise.
	This will cause noise in subsequent fits and 
	'real' freq may be misinterprted.
	This can produce false negative (A bias towards
	false positives has not been observed)
	* The up side to this is that non orthoganal 
	(or very close) freq are not all scored significantly.
	It is likley that the most 'important' is scored 
	significant while the highly dependent ones will
	not.
	"""
	if len(w)==0:
		w = defultFreq(t)
	_,m = x.shape
	n = len(w)
	Y = np.array(np.zeros((n,m)),dtype=bool)
	for i in range(m):
		p,_ = fourierSeries.estSigFreq(x[:,i],t,w,nPerm)
		Y[p<alpha,i] = True
	
	return Y,w
		






def defultFreq(t):
	"""Since we are not using uniform time points
	the standard harmonic frequancies do not make sense
	we have, somewhat arbitrarily, chosen the following
	w_min corrisponds to a maximum period 
	which contains all points will be the range of 
	max(t) - min(t), and the w_max corrisponds to a 
	minimum period that contains at worst 2 points, 
	on a sorrted t -> t_sort, max_n(t_n+1-t_n).
	we also include len(t)+1 diffrent uniformly 
	spaced w's, slight higher then we typically do
	to try to estimate a bit better, the statistical
	method used will account for non orthoganal 
	(very similar) freq.
	"""
	p_min = np.max(t[3:]-t[:-3])
	w_max = np.pi*2/p_min
	p_max = np.max(t)-np.min(t)
	w_min = np.pi*2/p_max
	w_step = (w_max-w_min)/len(t)
	w = np.arange(w_min,w_max+w_step,w_step)
	return w	



