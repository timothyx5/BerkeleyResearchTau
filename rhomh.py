"""
Metropolis-Hastings Algorithm

Authors: Nicholas Kern and Tim Wilson
"""

from __future__ import division
import sys
import numpy as np
import numpy.random as rd
import scipy.stats as stats
import numpy.linalg as la
import Tau
import lnL

n = np.load('/Users/TJW/BerkeleyResearchTau/ionHist.npz')

class Metro_Hast(object):

	def __init__(self,dic):
		self.__dict__.update(dic)

	def propose(self):
		prop_dist = self.prop(self.nwalkers)

		return self.walker_pos + prop_dist

	def accept_reject(self,proposals):
		# Define Alpha, iterate over walkers
		alpha = []
		for i in range(self.nwalkers):
			alpha.append( self.f(*proposals[i]) / self.f(*self.walker_pos[i]) )

		# Iterate through walkers
		for i in range(self.nwalkers):
			# Check proposal is within parameter bound
			within_bounds = True
			for j in range(self.dim):
				# Iterate through dimensions
				within_bounds *= (proposals[i][j] > self.param_bounds[j][0]) & (proposals[i][j] < self.param_bounds[j][1])
			# If within bounds, continue
			if within_bounds == True:
				# Accept or reject based on alpha
				if alpha[i] >= 1:
					self.walker_chain[i].append(1*self.walker_pos[i])
					self.walker_pos[i] = proposals[i]
				else:
					# Probabilistic acceptance even if alpha is > 1
					prob = alpha[i]
					rand_num = rd.random()
					if rand_num < prob:
						self.walker_chain[i].append(1*self.walker_pos[i])
						self.walker_pos[i] = proposals[i]
					else:
						pass
			else:
				pass

##############################################
################# RUN PROGRAM ################
##############################################

dim				= 4												# Number of dimensions of parameter space


param_bounds	= [ [ 0.0137 - 0.001*10, 0.0137 + 0.001*10],
		    	    [ 3.26 - 0.21*10, 3.26 + 0.21*10],
		    		[ 2.59 - 0.14*10, 2.59 + 0.14*10],
		    		[ 5.68 - 0.19*10, 5.68 + 0.19*10] ]										# Parameter Space Bounds

def f(ap=0.01376, bp=3.26, cp=2.59, dp=5.68):
	return -0.5*np.sum(((lnL.y - lnL.m(ap,bp,cp,dp))/lnL.sigma)**2) # Underlying distribution

sigma_prop 		= np.array([0.0001,0.03,0.02,0.05])
cov_prop		= np.eye(dim)*sigma_prop
mean_prop		= np.zeros(dim)
prop 			= lambda num: np.absolute(stats.multivariate_normal.rvs(mean=mean_prop,cov=cov_prop,size=num).reshape(dim,num).T)	# Proposal distribution


nwalkers		= 5 												# Number of walkers
walker_pos 	    = np.array([np.array([0.01376, 3.26, 2.59, 5.68])*i for i in np.random.randint(75,125,nwalkers)/100.0])
walker_chain	= [ [] for i in range(nwalkers) ]

# Stuff the variables together
variables = {'dim':dim,'param_bounds':param_bounds,'f':f,'prop':prop,'nwalkers':nwalkers,
				'walker_pos':walker_pos,'walker_chain':walker_chain}

## Initialize the class!
MH = Metro_Hast(variables)

N = 25000
for j in range(N):
	# Propose!
	proposals = MH.propose()

	# Accept / Reject
	MH.accept_reject(proposals)

	if j % 100 == 0: print j 
        sys.stdout.flush()


chain = np.array([ np.array(MH.walker_chain[i]) for i in range(nwalkers) ])

## Create master chain
master_chain = []
for k in range(nwalkers):
	master_chain.extend(chain[k])

master_chain = np.array(master_chain)

np.savez(sys.argv[1], master_chain=master_chain)
np.savez(sys.argv[2], chain=chain)
