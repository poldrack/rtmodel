"""
make figure showing RT modeling
"""

import numpy
import matplotlib.pyplot as plt
import scipy

def spm_hrf(TR,p=[6,16,1,1,6,0,32]):
    """ An implementation of spm_hrf.m from the SPM distribution

Arguments:

Required:
TR: repetition time at which to generate the HRF (in seconds)

Optional:
p: list with parameters of the two gamma functions:
                                                     defaults
                                                    (seconds)
   p[0] - delay of response (relative to onset)         6
   p[1] - delay of undershoot (relative to onset)      16
   p[2] - dispersion of response                        1
   p[3] - dispersion of undershoot                      1
   p[4] - ratio of response to undershoot               6
   p[5] - onset (seconds)                               0
   p[6] - length of kernel (seconds)                   32

"""

    p=[float(x) for x in p]

    fMRI_T = 16.0

    TR=float(TR)
    dt  = TR/fMRI_T
    u   = numpy.arange(p[6]/dt + 1) - p[5]/dt
    hrf=scipy.stats.gamma.pdf(u,p[0]/p[2],scale=1.0/(dt/p[2])) - scipy.stats.gamma.pdf(u,p[1]/p[3],scale=1.0/(dt/p[3]))/p[4]
    good_pts=numpy.array(range(numpy.int(p[6]/TR)))*fMRI_T
    hrf=hrf[list(good_pts)]
    # hrf = hrf([0:(p(7)/RT)]*fMRI_T + 1);
    hrf = hrf/numpy.sum(hrf);
    return hrf



tr=0.1

variable_sf=numpy.zeros(1000)

variable_sf[100:102]=1
variable_sf[300:303]=1
variable_sf[500:505]=1
variable_sf[700:706]=1
variable_sf_conv=numpy.convolve(variable_sf,spm_hrf(tr))[0:len(variable_sf)]

constant_sf=numpy.zeros(1000)
constant_sf[100:104]=1
constant_sf[300:304]=1
constant_sf[500:504]=1
constant_sf[700:704]=1
constant_sf_conv=numpy.convolve(constant_sf,spm_hrf(tr))[0:len(variable_sf)]

hrf_bases=numpy.zeros((1000,4))
hrf_bases[100:104,0]=1
hrf_bases[300:304,1]=1
hrf_bases[500:504,2]=1
hrf_bases[700:704,3]=1
desmtx=numpy.zeros((1000,4))

for x in range(4):
    desmtx[:,x]=numpy.convolve(hrf_bases[:,x],spm_hrf(tr))[0:len(variable_sf)]

b_est=numpy.linalg.inv(desmtx.T.dot(desmtx)).dot(desmtx.T).dot(variable_sf_conv)

intensity_sf=numpy.zeros(1000)
intensity_sf[100:104]=b_est[0]
intensity_sf[300:304]=b_est[1]
intensity_sf[500:504]=b_est[2]
intensity_sf[700:704]=b_est[3]
intensity_sf_conv=numpy.convolve(intensity_sf,spm_hrf(tr))[0:len(variable_sf)]

intensity_sf_conv=desmtx.dot(b_est)

plt.clf()
plt.plot(variable_sf_conv,color='k',linewidth=4)
plt.hold(True)
plt.plot(intensity_sf_conv,'c--')
#plt.plot(constant_sf_conv,color='b')
plt.plot(intensity_sf_conv - variable_sf_conv,color='b')
plt.text(10,-0.02,'RT')
plt.text(100,-0.02,'200 ms')
plt.text(300,-0.02,'300 ms')
plt.text(500,-0.02,'500 ms')
plt.text(700,-0.02,'600 ms')
plt.text(10,-0.03,'Beta')
plt.text(100,-0.03,'%0.2f'%b_est[0])
plt.text(300,-0.03,'%0.2f'%b_est[1])
plt.text(500,-0.03,'%0.2f'%b_est[2])
plt.text(700,-0.03,'%0.2f'%b_est[3])

plt.axis([0,1000,-0.05,0.15])
plt.legend(['Variable duration','Variable intensity (fitted)','Difference'],loc='upper left')
plt.savefig('figure1.pdf')

