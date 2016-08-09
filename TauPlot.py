import matplotlib.pyplot as plt
import numpy as np
import Tau

n = np.load('c:/Users/Timot/Desktop/Research/ionHist.npz')

Qn = np.ones(len(Tau.Q_adrian)) - Tau.Q_adrian

###################################################
################## Plotting #######################
###################################################

plt.semilogy(Tau.z, Tau.rho(Tau.z), label='Planck Results', color='red')
plt.semilogy(Tau.z, Tau.rho(Tau.z, 0.01306, 3.66, 2.28, 5.29), label=r'Forced Match to WMAP  $\tau$', color='orange')
plt.semilogy(Tau.z, Tau.rho(Tau.z, 0.01467, 3.279, 2.988, 5.515), label=r'MCMC MEAN $\tau$', color='blue')
plt.semilogy(Tau.z, Tau.rho(Tau.z, 0.0180, 3.87, 2.65, 5.89), label=r'MCMC MEDIAN $\tau$', color='green')
plt.errorbar(Tau.uv_z, Tau.uv_data, yerr=[Tau.uv_err_low, Tau.uv_err_up], fmt='o', color='blue')
plt.errorbar(Tau.ir_z, Tau.ir_data, yerr=[Tau.ir_err_low, Tau.ir_err_up], fmt='o', color='red')
#plt.axis([0,15,10**-3.5,10**-0.5])
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)
plt.legend(frameon=0, loc='lower left')
plt.grid(True)
plt.xlabel(r'Redshift $z$',fontsize=28)
plt.ylabel(r'$log_{10}$ $\rho_{SFR} \ [M_{\odot} yr^{-1} Mpc^{-3}]$',fontsize=28)

plt.show()

plt.plot(n['base_zArray'],n['nf0'], color='black', label='21CMFAST simulation')
plt.fill_between(n['base_zArray'],n['lowerCurve95'],n['upperCurve95'], color='grey')
plt.plot(n['base_zArray'], Qn, label='model data')
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)
plt.legend(frameon=0, loc='lower right')
plt.ylabel(r'Neutral Hydrogen Fraction $1 - Q_{HII}$', fontsize=28)
plt.xlabel(r'Redshift $z$', fontsize=28)

plt.plot(Tau.z[0:-1], Tau.tau) # z[0:-1] to match the array dimensions
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)
plt.grid(True)
plt.xlabel(r'Redshift $z$',fontsize=28)
plt.ylabel(r'Thomson Optical Depth $\tau$',fontsize=28)
