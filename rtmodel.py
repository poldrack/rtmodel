"""
make figure showing RT modeling
"""

import numpy
import spm_hrf
import matplotlib.pyplot as plt

import statsmodels.nonparametric.smoothers_lowess
tr=0.1

sf=numpy.zeros(1000)

sf[100:102]=1
sf[300:303]=1
sf[500:505]=1
sf[700:706]=1

intensity_sf=numpy.zeros(1000)

intensity_sf[100:104]=0.4
intensity_sf[300:304]=0.8
intensity_sf[500:504]=1.2
intensity_sf[700:704]=1.6

constant_sf=numpy.zeros(1000)

constant_sf[100:104]=1
constant_sf[300:304]=1
constant_sf[500:504]=1
constant_sf[700:704]=1

noise=statsmodels.nonparametric.smoothers_lowess.lowess(numpy.random.randn(len(sf))*0.03,numpy.arange(len(sf)),0.1)[:,1]
hrf=numpy.convolve(sf,spm_hrf.spm_hrf(tr))[0:len(sf)]
hrf_int=numpy.convolve(intensity_sf,spm_hrf.spm_hrf(tr))[0:len(sf)]
constant_sf_conv=numpy.convolve(constant_sf,spm_hrf.spm_hrf(tr))[0:len(sf)]
convdata = hrf # + noise
plt.plot(hrf,'b')
plt.hold(True)
plt.plot(hrf_int,'g')
plt.axis([0,1000,-0.05,0.15])

plt.savefig('rtmodel_data.pdf')


plt.hold(True)
plt.plot(hrf,'red')
plt.savefig('rtmodel_grinband.pdf')

plt.clf()
plt.plot(convdata,'b')
plt.axis([0,1000,-0.05,0.15])
plt.plot(constant_sf_conv,'r')
plt.savefig('rtmodel_constant.pdf')

plt.hold(True)
plt.plot(hrf - constant_sf_conv,'g')

plt.savefig('rtmodel_constant_plus_param.pdf')
plt.clf()

plt.plot(sf*0.1,'k')
plt.axis([0,1000,-0.05,0.15])
plt.savefig('rtmodel_stick.pdf')
plt.hold(True)

plt.plot(hrf,'b')
plt.savefig('rtmodel_stick_with_hrf.pdf')
