import pymultinest
import spotrod
import numpy as np
try:
    import matplotlib
    ShowPlots = True
except:
    print 'No matplotlib, so no final plots!'
    ShowPlots = False
import pickle
import os

############ OPTIONAL INPUTS ###################
# Name of the file containing the data (first column time, second column relative flux)
filename = 'my_lightcurve.dat'
# Number of spots in the model. 0: no spot fit, 1: one spot, 2: two spots
nspots = 1

# Number of MultiNest live points:
n_live_points = 1000
# Around which phases do you want to fit the transit + spot model?
PHASE_FIT = 0.02

# Resampling? Set it to yes if using Kepler long-cadence data:
RESAMPLING = False
NRESAMPLING = 20

# Set the noise level of your data in ppm:
sigma_w = 2000.0 # ppm

# Transit model priors:
rp = 0.1                                 # Rp/Rs
rp_sigma = (0.0011563461+0.0016538642)/2.
aR = 6.0                                 # a/Rs
aR_sigma = (0.3644584151+0.3414799136)/2.
inc = 87.0                               # Inclination
inc_sigma = (0.6364921359+0.7092897909)/2.  
t0 = 2458028.0                           # Time-of-transit center
t0_sigma = (0.0002299240+0.0002043541)/2.
ld_law = 'quadratic'                              # Limb-darkening law (quadratic supported by now, easy to implement other laws).
###############################################


# Fixed components:
t,flux = np.loadtxt(filename,unpack=True)
ndata = len(t)
P = 6.0
ecc = 0.3
omega = 60.0
deg_to_rad = (np.pi/2.)/90.
k = ecc*np.cos(omega*deg_to_rad)
h = ecc*np.sin(omega*deg_to_rad)

flux_err = sigma_w*1e-6
ntransit = 0
def get_phases(t,P,t0):
    """ 
    Given input times, a period (or posterior dist of periods)
    and time of transit center (or posterior), returns the 
    phase at each time t.
    """
    if type(t) is not float:
        phase = ((t - np.median(t0))/np.median(P)) % 1 
        ii = np.where(phase>=0.5)[0]
        phase[ii] = phase[ii]-1.0
    else:
        phase = ((t - np.median(t0))/np.median(P)) % 1 
        if phase>=0.5:
            phase = phase - 1.0 
    return phase

# Remove out-of-transit points:
phases = get_phases(t,P,t0)
idx_fit = np.where(np.abs(phases)<PHASE_FIT)[0]

from scipy.stats import norm,beta
def transform_uniform(x,a,b):
    return a + (b-a)*x

def transform_loguniform(x,a,b):
    la=np.log(a)
    lb=np.log(b)
    return np.exp(la + x*(lb-la))

def transform_normal(x,mu,sigma):
    return norm.ppf(x,loc=mu,scale=sigma)

def transform_beta(x,a,b):
    return beta.ppf(x,a,b)

def prior(cube, ndim, nparams):
    # Prior on "rp" is gaussian:
    cube[0] = transform_normal(cube[0],rp,rp_sigma)
    # Same for aR:
    cube[1] = transform_normal(cube[1],aR,aR_sigma)
    # And inc:
    cube[2] = transform_normal(cube[2],inc,inc_sigma)
    # And t0:
    cube[3] = transform_normal(cube[3],t0,t0_sigma)
    # Uniform on the transformed LD coeffs (q1 and q2):
    cube[4] = transform_uniform(cube[4],0,1)
    cube[5] = transform_uniform(cube[5],0,1)
    if nspots == 1 or nspots == 2:
        # And now uniform in position of spot,
        cube[6] = transform_uniform(cube[6],-1,1)
        cube[7] = transform_uniform(cube[7],-1,1)
        # And also in radius of the spot:
        cube[8] = transform_uniform(cube[8],0,1)
        # And contrast:
        cube[9] = transform_uniform(cube[9],-2,2)
    if nspots == 2:
        # And now uniform in position of spot,
        cube[10] = transform_uniform(cube[10],-1,1)
        cube[11] = transform_uniform(cube[11],-1,1)
        # And also in radius of the spot:
        cube[12] = transform_uniform(cube[12],0,1)
        # And contrast:
        cube[13] = transform_uniform(cube[13],-2,2)

def quadraticlimbdarkening(r, s1, s2):
  answer = np.zeros_like(r)
  mask = (r<=1.0)
  mu = np.sqrt(1.0 - np.power(r[mask],2))
  answer[mask] = 1.0 - s1 * (1.-mu) - s2 * (1.-mu)**2
  return answer

def reverse_ld_coeffs(ld_law, q1, q2): 
    if ld_law == 'quadratic':
        coeff1 = 2.*np.sqrt(q1)*q2
        coeff2 = np.sqrt(q1)*(1.-2.*q2)
    elif ld_law=='squareroot':
        coeff1 = np.sqrt(q1)*(1.-2.*q2)
        coeff2 = 2.*np.sqrt(q1)*q2
    elif ld_law=='logarithmic':
        coeff1 = 1.-np.sqrt(q1)*q2
        coeff2 = 1.-np.sqrt(q1)
    return coeff1,coeff2

def spot_model(t,cube,nspot,texp = 0.01881944,RESAMPLE = False,NRESAMPLE=20):
    if RESAMPLE:
        tin = np.array([])
        for i in range(len(t)):
            for j in range(NRESAMPLE):
                jj = j+1
                tin = np.append(tin,t[i] + (jj - (NRESAMPLE + 1)/2.)*(texp/NRESAMPLE))
    else:
        tin = t
    # Extract parameters:
    if nspot == 0:
        rp,aR,inc,t0,q1,q2 = cube[0],cube[1],cube[2],cube[3],cube[4],cube[5]
    elif nspot == 1:
        rp,aR,inc,t0,q1,q2,spotx,spoty,spotradius,spotcontrast = cube[0],cube[1],\
                   cube[2],cube[3],cube[4],cube[5],cube[6],cube[7],cube[8],cube[9]
    elif nspot == 2:
        rp,aR,inc,t0,q1,q2,spotx,spoty,spotradius,spotcontrast,spotx2,spoty2,spotradius2,\
        spotcontrast2 = cube[0],cube[1],cube[2],cube[3],cube[4],cube[5],cube[6],cube[7],cube[8],\
                        cube[9],cube[10],cube[11],cube[12],cube[13]
    # Generate model:
    impactparam = aR*np.cos(inc*deg_to_rad)
    phase = np.mod((tin-t0)/P+0.5, 1.0)-0.5
    s1,s2 = reverse_ld_coeffs(ld_law,q1,q2)
    # Initialize spotrod. Number of intergration rings:
    n = 1500
    # Integration annulii radii.
    r = np.linspace(1.0/(2*n), 1.0-1.0/(2*n), n)
    # Weights: 2.0 times limb darkening times width of integration annulii.
    f = 2.0 * quadraticlimbdarkening(r, s1, s2) / n
    # Calculate orbital elements.
    eta, xi = spotrod.elements(tin-t0, P, aR, k, h)
    planetx = impactparam*eta/aR
    planety = -xi
    z = np.sqrt(np.power(planetx,2) + np.power(planety,2))
    planetangle = np.array([spotrod.circleangle(r, rp, z[i]) for i in xrange(z.shape[0])])

    if nspot == 0:
        spotx,spoty,spotradius,spotcontrast = 0.,0.,0.,1.
        model = spotrod.integratetransit(planetx, planety, z, rp, r, f, np.array([spotx]),\
                np.array([spoty]), np.array([spotradius]), np.array([spotcontrast]), planetangle)
    elif nspot == 1:
        model = spotrod.integratetransit(planetx, planety, z, rp, r, f, np.array([spotx]),\
                np.array([spoty]), np.array([spotradius]), np.array([spotcontrast]), planetangle)
    elif nspot == 2:
        model = spotrod.integratetransit(planetx, planety, z, rp, r, f, np.array([spotx,spotx2]),\
                np.array([spoty,spoty2]), np.array([spotradius,spotradius2]), np.array([spotcontrast,spotcontrast2]), planetangle)
    if RESAMPLE:
        model_out = np.zeros(len(t))
        for i in range(len(t)):
            model_out[i] = np.mean(model[i*NRESAMPLE:NRESAMPLE*(i+1)])
        return model_out
    return model

def loglike(cube, ndim, nparams):
    model = spot_model(t[idx_fit],cube,nspots,RESAMPLE=RESAMPLING,NRESAMPLE=NRESAMPLING)
    # Evaluate the log-likelihood:
    loglikelihood = -0.5*ndata*np.log(2.*np.pi*flux_err**2) + (-0.5 * ((model - flux[idx_fit]) / flux_err)**2).sum()
    return loglikelihood

n_params = 6
if nspots == 1:
    n_params = n_params + 4
elif nspots == 2:
    n_params = n_params + 8

out_file = 'out_multinest_transit_'+str(nspots)
out_pickle_name = 'POSTERIOR_SAMPLES_nspot_'+str(nspots)+'.pkl'

if not os.path.exists(out_pickle_name):
    # Run MultiNest:
    pymultinest.run(loglike, prior, n_params, n_live_points = n_live_points,outputfiles_basename=out_file, resume = False, verbose = True)
    # Get output:
    output = pymultinest.Analyzer(outputfiles_basename=out_file, n_params = n_params)
    # Get out parameters: this matrix has (samples,n_params+1):
    
    mc_samples = output.get_equal_weighted_posterior()[:,:-1]
    a_lnZ = output.get_stats()['global evidence']
    logZ = (a_lnZ / np.log(10))
    out = {}
    out['posterior_samples'] = mc_samples
    out['logZ'] = (a_lnZ / np.log(10))
    pickle.dump(out,open(out_pickle_name,'wb'))
else:
    mc_samples = pickle.load(open(out_pickle_name,'rb'))['posterior_samples']

if ShowPlots:
    matplotlib.pyplot.style.use('ggplot')
    tmodel = np.linspace(np.min(t),np.max(t),1000)
    matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Arial']})
    #matplotlib.rc('text', usetex=True)
    matplotlib.rcParams.update({'font.size':12})
    matplotlib.pyplot.errorbar((t-t0)*24.,flux,yerr=np.ones(len(t))*flux_err,fmt='.',color='black')
    nsample = 500
    idx = np.random.choice(mc_samples.shape[0],nsample,replace=False)
    for i in range(nsample):
        out_cube = mc_samples[idx[i],:]
        matplotlib.pyplot.plot((tmodel-t0)*24.,spot_model(tmodel,out_cube,nspots,RESAMPLE=RESAMPLING,NRESAMPLE=NRESAMPLING),'r-',alpha=0.05)
    matplotlib.pyplot.ylabel('Relative flux')
    matplotlib.pyplot.xlabel('Time from transit center (hours)')
    matplotlib.pyplot.show()
print 'Done!'
