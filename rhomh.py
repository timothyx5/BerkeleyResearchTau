"""
Metropolis-Hastings Algorithm
"""

from __future__ import division
import numpy as np
import numpy.random as rd
import scipy.stats as stats
import numpy.linalg as la
import matplotlib.pyplot as plt
import Tau
import lnL

np.load('c:/Users/Timot/Downloads/ionHist.npz')
n = np.load('c:/Users/Timot/Downloads/ionHist.npz')

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

param_bounds	= [	[-0.1, 0.1] ,
					[-32.6, 32.6],
					[0.001, 25.9],
					[-56.8, 56.8] ]									# Parameter Space Bounds

def f(ap=0.01376, bp=3.26, cp=2.59, dp=5.68):
	return -1./2*np.sum(((lnL.y - lnL.m(ap,bp,cp,dp))/lnL.sigma)**2) # Underlying distribution

sigma_prop 		= np.array([0.0001,0.03,0.03,0.05])
cov_prop		= np.eye(dim)*sigma_prop
mean_prop		= np.zeros(dim)
prop 			= lambda num: stats.multivariate_normal.rvs(mean=mean_prop,cov=cov_prop,size=num).reshape(dim,num).T	# Proposal distribution


nwalkers		= 5 												# Number of walkers
#walker_pos		= np.array([stats.uniform.rvs(param_bounds[i][0],param_bounds[i][1]-param_bounds[i][0],nwalkers) for i in range(dim)]).T			# (leftstart, length, number)
walker_pos 	    = np.array([np.array([0.01376, 3.26, 2.59, 5.68])*i for i in np.random.randint(0,50,nwalkers)/200])
walker_chain	= [ [] for i in range(nwalkers) ]

# Stuff the variables together
variables = {'dim':dim,'param_bounds':param_bounds,'f':f,'prop':prop,'nwalkers':nwalkers,
				'walker_pos':walker_pos,'walker_chain':walker_chain}

## Initialize the class!
MH = Metro_Hast(variables)

N = 5000
for j in range(N):
	# Propose!
	proposals = MH.propose()

	# Accept / Reject
	MH.accept_reject(proposals)

	if j % 100 == 0: print j


chain = np.array([ np.array(MH.walker_chain[i]) for i in range(nwalkers) ])

## Create master chain
master_chain = []
for k in range(nwalkers):
	master_chain.extend(chain[k])

master_chain = np.array(master_chain)


## Plotting a histogram of one dimesion ##
plot1 = False
if plot1 == True:

	# Choose 1 dimension to plot
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.hist(master_chain.T[0], bins=25, color='g', histtype='step',alpha=0.25,range=(-10,10))

# Choose 1 dimension to vary, and hold all others constant
plot_dim = 0

# Initialize x vector, then iterate over 
x_array = []
for k in np.linspace(param_bounds[plot_dim][0],param_bounds[plot_dim][1],500):
	x_vec = np.zeros(dim)
	x_vec[plot_dim] = k*1
	x_array.append(x_vec)

x_array = np.array(x_array)

plot2 = False
if plot2 == True:
	## Plotting the Underlying f function across one dimension ##

	# Evaluate the function f over vector x_array
	f_array = f(x_array)

	# Plot
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(x_array.T[plot_dim],f_array,color='b',linewidth=2,alpha=0.5)


# Initialize x vector, then iterate over 
x_array = []
for k in np.linspace(param_bounds[plot_dim][0],param_bounds[plot_dim][1],500):
	x_vec = np.zeros(dim)
	x_vec[plot_dim] = k*1
	x_array.append(x_vec)

#x_array = np.array(x_array)
#f_array = f(x_array)

plot3 = False
if plot3 == True:
	## Plotting the Underlying f function AND the histogram of the master_chain across one dimension ##

	# Get chain samples of conditional distribution at theta=0
	select = np.where( (np.abs(master_chain.T[1]-0)<1))# & (np.abs(master_chain.T[1]-0)<1) )
	conditional_chain = master_chain[select]

	# Initialize subplot
	fig = plt.figure()
	ax = fig.add_subplot(111)
	# Plot histogram
	hist_data = ax.hist(conditional_chain.T[plot_dim],bins=50,range=param_bounds[plot_dim],color='r',histtype='step')
	# Get histogram maximum
	hist_max = np.max(hist_data[0])
	# plot function f, normalized by hist_max
	ax.plot(x_array.T[plot_dim],f_array*(hist_max/np.max(f_array)),color='b',linewidth=2,alpha=0.5)


plot4 = False
if plot4 == True:
	## Plotting 2 parameters ##

	# Make meshgrid
	XX,YY = np.meshgrid(np.linspace(param_bounds[0][0],param_bounds[0][1],25),
					np.linspace(param_bounds[1][0],param_bounds[1][1],25))

	# Reshape from matrix to row vector
	Z = np.vstack([XX.ravel(),YY.ravel()]).T

	# Evaluate f at meshgrid points
	f_array = f(Z)

	# Get Kernel Density Estimate of samples
	kde = stats.gaussian_kde(master_chain.T)
	chain_2dhist = kde(Z.T).reshape(25,25)

	# Initialize plot
	fig = plt.figure()
	ax = fig.add_subplot(111)

	# plot heat map of function f, which has been reshaped back to matrix
	ax.imshow(f_array.reshape(25,25), extent=[-10,10,-10,10],origin='lower',cmap='magma')

	# get contours from data
	ax.contour(chain_2dhist, extent=[-10,10,-10,10])

#plt.scatter(master_chain.T[0], master_chain.T[1], alpha=0.5, s=1)

#plt.show()