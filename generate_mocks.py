# import standard python modules
import numpy as np
import time
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
import matplotlib.pyplot as plt
from lenstronomy.Util import param_util 
from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions
from matplotlib.patches import Ellipse
from matplotlib.ticker import NullFormatter
#from scipy import stats
import json
import cosmolopy.distance as cd
import random
import os
from scipy.special import gamma as gammafunc
from scipy.interpolate import interp1d
from scipy.interpolate import RectBivariateSpline
import pickle
#z_lens = 0.599
#z_source = 2.517

#cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.)
#lensCosmo = LensCosmo(cosmo=cosmo, z_lens=z_lens, z_source=z_source)


lens_model_list = ['SIE', 'SHEAR']
#kwargs_shear = {'gamma_ext': 0.05, 'psi_ext': 1.8}

def distance_angles(x_image, y_image):
    distances = (x_image**2. + y_image**2)**0.5
    v1 = np.array([x_image[0], y_image[0]])
    v2 = np.array([x_image[1], y_image[1]])

    v1 /= (np.array(v1)**2).sum()**0.5
    v2 /= (np.array(v2)**2).sum()**0.5

    angle = np.arccos(np.dot(v1, v2))*180./np.pi

    return np.sort(distances)[::-1], angle

h = 0.7
alpha = -3.31
alpha2 = -2.58
beta = -1.45
phistar = 5.34*(10**-6)*h**3
cosmo = {'omega_M_0':0.3089, 'omega_lambda_0':0.6911, 'omega_k_0':0.0, 'h':h}
cosmo = {'omega_M_0':0.26, 'omega_lambda_0':0.74, 'omega_k_0':0.0, 'h':0.72}
kcorr = np.genfromtxt('./i_band_k_corr.csv', dtype=float).T[1]


def make_dS_spline(zmin=0., zmax=5.5, dz=0.001, savename='./dS_spline.npy'):
    """Make an angular diameter distance spline for the source redshift 
    """
    zs = np.arange(zmin, zmax+dz, dz)
    dSs = []
    for z in zs:
        dS = cd.angular_diameter_distance(z, z0=0., **cosmo)
        dSs.append(dS)

    spl = interp1d(zs, dSs, kind='cubic')
    with open(savename, 'wb') as file:
        pickle.dump(spl, file)


def make_ddS_spline(zmin=0., zmax=5.5, dz=0.001, savename='./ddS_spline.npy'):
    """Make an angular diameter distance bivariate spline for the lens-source redshift 
    """
    zs = np.arange(zmin, zmax+dz, dz)
    zs = np.arange(zmin, zmax+dz, dz)

    ddSs = []
    for z in zs:
        ddS = cd.angular_diameter_distance(zs, z0=z, **cosmo)
        ddSs.append(ddS)

    values = np.array(ddSs).T
    spl = RectBivariateSpline(zs, zs, values)
    with open(savename, 'wb') as file:
        pickle.dump(spl, file)


def f(z, zeta=2.98, xi=4.05, zstar=1.60):
    """See OM10 for details
    """
    f = np.exp(zeta*z)*(1+np.exp(xi*zstar))/((np.exp(xi*z/2.)+np.exp(xi*zstar/2))**2.)
    return f

def phi(z, M, h, alpha, alpha2, beta, phistar):
    """See OM10 for details
    """
    if z > 3:
        alpha = alpha2
    
    Mstar = -20.90 + 5*np.log10(h) -2.5*np.log10(f(z))
    dphidm = phistar/(10**(0.4*(alpha+1)*(M-Mstar)) + 10**(0.4*(beta+1)*(M-Mstar)))
    return dphidm

def make_redshift_pdf(zmin=0., zmax=5., Mmin=-28, Mmax=-21, dz=0.001, dM=0.01, savename='./redshift_cdf.npy'):
    """Inntegrate over the source luminosity function at each redshift, to get a probability distribution
    function as a function of redshift. This can then be used to draw a redshift (see draw_redshifts)
    """
    integrals = []
    for z in np.arange(zmin, zmax, dz):
        integral = 0
        for M in np.arange(Mmin, Mmax, dM):
            comovingvolumez1z2 = cd.diff_comoving_volume(z+(dz/2.), **cosmo)*dz
            integral += dM*comovingvolumez1z2*phi(z, M, h, alpha, alpha2, beta, phistar)
        print(integral)
        integrals.append(integral)

    np.save(savename, np.array([np.array(integrals)/np.sum(integrals), np.arange(zmin, zmax, dz)]))

def draw_redshifts(N=1):
    """Draw a source redshift from the 1d redshift PDF
    """
    redshifts = random.choices(zsources, weights=probs_zsources, k=N)

    if N==1:
        redshifts = redshifts[0]
    
    return redshifts


def draw_M(z, Mmin=-28, Mmax=-21, dM=0.01, N=1):
    """Draw a source luminosity from the luminosity function *given* a source redshift
    """    
    dphidm = []
    for M in np.arange(Mmin, Mmax, dM):
        dphidm.append(phi(z, M, h, alpha, alpha2, beta, phistar))
    M = random.choices(np.arange(Mmin, Mmax, dM), weights=dphidm, k=N)

    if N==1:
        M = M[0]

    return M


def obs_imag(z, M):
    """Convert the source luminosity to observed magnitude, see OM10 for details
    """
    dL = cd.luminosity_distance(z,**cosmo) 
    
    #now we need to convert to apparent magnitudes (so we need a distance modulus and a K-correction)
    z_kcorr = np.arange(kcorr.shape[0])*0.01
    index = np.argmin(np.abs(z-z_kcorr))
    Kcorr = kcorr[index]
    m = M + 5.*np.log10(dL*10**5.) + Kcorr
    #plt.plot(z_kcorr, kcorr, '+')
    #plt.show()

    return m


def draw_source(imagmax=28, zmin=0., zmax=5., Mmin=-28, Mmax=-21, dz=0.01, dM=0.1):
    """Draw a source quasar
    """
    #the true distribution is the luminosity function phi(M, z) multiplied by the comoving infinitescimal volume element
   
    imag = 1000
    while imag>imagmax:
        parstring = 'dz_'+str(dz)+'_dM_'+str(dM)+'_zmin_'+str(zmin)+'_zmax_'+str(zmax)+'_Mmin_'+str(Mmin)+'_Mmax_'+str(Mmax)
        z = draw_redshifts(N=1, zmin=zmin, zmax=zmax, Mmin=Mmin, Mmax=Mmax, dz=dz, dM=dM, savename='./redshift_cdf_'+parstring+'.npy')
        M = draw_M(z=z, Mmin=Mmin, Mmax=Mmax, N=1)
        imag = obs_imag(z, M)
    return imag, z, M


def draw_source_givenz(z, imagmax=28, Mmin=-28, Mmax=-21, dM=0.1):
    """Draw a source quasar given a redshift
    """
    #the true distribution is the luminosity function phi(M, z) multiplied by the comoving infinitesimal volume element
   
    #imag = 1000
    #while imag>imagmax:
    M = draw_M(z=z, Mmin=Mmin, Mmax=Mmax, N=1)
    imag = obs_imag(z, M)
    return imag, M

def make_lens_redshift_pdf(zmin=0., zmax=2.5, dz=0.001, savename='./lens_redshift_pdf.npy'):
    """PDF for the lens redshift
    """
    integrals = []
    for z in np.arange(zmin, zmax, dz):
        comovingvolumez1z2 = cd.diff_comoving_volume(z+(dz/2.), **cosmo)*dz
        integrals.append(comovingvolumez1z2)

    np.save(savename, np.array([np.array(integrals)/np.sum(integrals), np.arange(zmin, zmax, dz)]))



def draw_lens_redshift(N=1):
    """Draw a lens redshift
    """
    z = random.choices(zs, weights=probs_zs, k=N)

    if N==1:
        z = z[0]
    return z

def phi_vdisp(v):
    """Lens velocity dispersion function, as in OM10
    """
    phi_vdisp_star = 0.008*(h**3)# = (8*10**-3)*(h**3)
    vstar = 161.0
    alpha_vdisp = 2.32
    beta_vdisp = 2.67
    gammaab = gammafunc(alpha_vdisp/beta_vdisp)
    dndv = phi_vdisp_star*((v/vstar)**alpha_vdisp)*(np.exp(-((v/vstar)**beta_vdisp)))*(beta_vdisp/gammaab)/v

    return dndv #*v*np.log(10)

def make_vdisp_pdf(sigmamin=70, sigmamax=350, dsigma=0.1, savename='./lens_sigma_pdf.npy'):
    """Make PDF for Lens velocity dispersion function
    """
    integrals = []
    for sigma in np.arange(sigmamin, sigmamax, dsigma):
        phi = phi_vdisp(sigma)
        integrals.append(phi)

    np.save(savename, np.array([np.array(integrals)/np.sum(integrals), np.arange(sigmamin, sigmamax, dsigma)]))


def draw_lens_sigma(N=1):
    """Draw just a lens velocity dispersion (note here there is no redshift evolution)
    """
    sig = random.choices(sigmas, weights=probs_sigmas, k=N)

    if N==1:
        sig = sig[0]

    return sig

def draw_lens(N=1):
    """Draw a lens
    """
    z = draw_lens_redshift(N=N)
    sig = draw_lens_sigma(N=N)
    return z, sig


'''def multiply_imaged(theta, ra, dec):
    qval = np.random.normal(0.7, 0.16)
    if qval > 1:
        qval = 1
    elif qval < 0.1:
        qval = 0.1
    phi_e = np.random.uniform(0, np.pi)
    e1, e2 = param_util.phi_q2_ellipticity(phi=phi_e, q=qval)
    gamma = np.random.lognormal(mean=np.log(0.05), sigma=0.2*np.log(10))
    gamma_phi = np.random.uniform(0, 2.*np.pi)
    gamma1, gamma2 = param_util.shear_polar2cartesian(phi=gamma_phi, gamma=gamma)
    kwargs_shear = {'gamma1': gamma1, 'gamma2': gamma2, 'ra_0': 0, 'dec_0': 0}
    kwargs_spemd = {'theta_E': theta, 'center_x': 0, 'center_y': 0, 'e1': e1, 'e2': e2}
    kwargs_lens = [kwargs_spemd, kwargs_shear]
    lens_model_class = LensModel(lens_model_list=lens_model_list)
    lensEquationSolver = LensEquationSolver(lens_model_class)
    x_image, y_image = lensEquationSolver.findBrightImage(ra, dec, kwargs_lens, min_distance=0.005, search_window=5., num_iter_max=20)
    if len(x_image)>1:
        return True
    else:
        return False'''

'''def setup_sourcedistance_theta_relation():
    thetas = np.random.uniform(0, 5, 10000)
    ras = np.random.uniform(0, 2, 10000)
    decs = np.random.uniform(0, 2, 10000)
    lensed = []
    for i in range(len(ras)):
        print(i)
        lensed.append(multiply_imaged(thetas[i], ras[i], decs[i]))
    lensed = np.array(lensed)
    distances = (ras**2. + decs**2.)**0.5
    plt.scatter(thetas[lensed], distances[lensed])
    plt.xlim(0, 5)
    plt.ylim(0, 5)
    plt.xlabel('einstein radius (arcsec)')
    plt.ylabel('source distance (arcsec)')
    plt.show()'''



#def max_source_distance_from_lens(theta):
'''
This function allows one to know immediately if a source will be multiply imaged by an SIE+shear instead of having to solve the lens equation
param theta float, in arcseconds
'''


def plot_lens(ax, x_image, y_image, phi_e, qval, gamma, gamma_phi, mag, x_source, y_source):
    """
    Plot a cartoon version of the lens on ax
    """
    ax.plot(0, 0, marker='+', color='orange')
    ax.scatter(x_image, y_image, marker='o', s=100*(mag**0.5), edgecolor='blue', facecolors='none')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect(True)
    ax.set_xticks([], [])
    ax.set_yticks([], [])
    #print(qval, 180.*phi_e/np.pi)
    ellipse = Ellipse((0, 0), width=1., height=qval, angle=180.*phi_e/np.pi, edgecolor='orange', facecolor='none')
    ax.add_patch(ellipse)
    #print(gamma, gamma_phi*180./np.pi)
    ax.plot(ra_source, dec_source, '+', color='red', markersize=15)
    ax.arrow(0, 0, 3*gamma*np.cos(gamma_phi), 3*gamma*np.sin(gamma_phi), head_width=0.05, head_length=0.1, fc='k', ec='k')
    #distances, angle = distance_angles(x_image, y_image)
    #ax.text(0.5, 0.9, r'$\theta=$'+str(int(angle)), horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)




#load or make the interpolated angular diameter distance functions
ds_spline_file = './dS_spline.pkl'
if not os.path.exists(ds_spline_file):
    dS_spline = make_dS_spline(zmin=0., zmax=5.5, dz=0.01, savename=ds_spline_file)

with open(ds_spline_file, 'rb') as file:
    dS_spline = pickle.load(file)


dds_spline_file = './ddS_spline.pkl'
if not os.path.exists(dds_spline_file):
    make_ddS_spline(zmin=0., zmax=5.5, dz=0.01, savename=dds_spline_file)

with open(dds_spline_file, 'rb') as file:
    ddS_spline = pickle.load(file)

#define the velocity dispersion and redshift ranges of the lens
sigmamin, sigmamax, dsigma = 100, 410, 0.1
zmin, zmax, dz = 0., 2.5, 0.001

#make the PDF functions to draw from
parstring = 'dsigma_'+str(dsigma)+'_sigmamin_'+str(sigmamin)+'_sigmamax_'+str(sigmamax)
if not os.path.exists('./lens_sigma_pdf'+parstring+'.npy'):
    make_vdisp_pdf(sigmamin=sigmamin, sigmamax=sigmamax, dsigma=dsigma, savename='./lens_sigma_pdf'+parstring+'.npy')
probs_sigmas, sigmas = np.load('./lens_sigma_pdf'+parstring+'.npy')


parstring = 'dz_'+str(dz)+'_zmin_'+str(zmin)+'_zmax_'+str(zmax)
if not os.path.exists('./lens_redshift_cdf_'+parstring+'.npy'):
    make_lens_redshift_pdf(zmin=zmin, zmax=zmax, dz=dz, savename='./lens_redshift_cdf_'+parstring+'.npy')
probs_zs, zs = np.load('./lens_redshift_cdf_'+parstring+'.npy')


#define source magnitude and redshift ranges
zmin, zmax, dz = 0., 5.5, 0.01
Mmin, Mmax, dM = -28.0, -20.5, 0.05
parstring = 'dz_'+str(dz)+'_dM_'+str(dM)+'_zmin_'+str(zmin)+'_zmax_'+str(zmax)+'_Mmin_'+str(Mmin)+'_Mmax_'+str(Mmax)
savename='./redshift_cdf_'+parstring+'.npy'
print(savename)

if not os.path.exists(savename):
    make_redshift_pdf(zmin=zmin, zmax=zmax, Mmin=Mmin, Mmax=Mmax, dz=dz, dM=dM, savename=savename)

probs_zsources, zsources = np.load(savename)




#MAKE THE LENSES
N, M = 2, 5 #y, x
nlenses = 0
qmean = 0.7
qstd = 0.16
c = 299792. #km/s
fig = plt.figure(figsize=(M, N))
#final_lenses = []

filename = 'lenses.npy'
if os.path.exists(filename):
    final_lenses = list(np.load(filename, allow_pickle=True))
else:
    final_lenses = []

thrown_out_doubles = 0
todist = []
ntries = 0
while nlenses<N*M:
    ntries += 1
    #time to draw a lens is ~0.0007 seconds
    zlens, sigmalens = draw_lens(N=1)
    #time to draw a source redshift is ~0.001 seconds
    zsource = draw_redshifts(N=1)

    #if the lens is further than the source, it's not the type of lens we're looking for    
    if zsource<zlens:
        continue
    
    #D_d = cd.angular_diameter_distance(zlens, z0=0., **cosmo)
    #time for spline sample is 0.000055 seconds; time for calling cd.angular... is 0.00028 seconds
    D_s = dS_spline(zsource)
    
    #D_ds = cd.angular_diameter_distance(zsource, z0=zlens, **cosmo)[0]
    #time for spline sample is 0.0001 seconds; time for calling cd.angular... is slower
    D_ds = ddS_spline(zsource, zlens)[0][0]
    
    #calculate the Einstein radius in arcseconds
    theta = 4.*np.pi*((sigmalens/c)**2.)*D_ds/D_s
    theta *= (180./np.pi)*3600.
    
    #no need to consider it if <0.2 since we are sure it won't give image separation > 0.5
    if theta<0.2:
        continue

    #draw a random source and calculate distance from axis
    ra_source, dec_source = np.random.uniform(-5., 5.), np.random.uniform(-5., 5.)
    dist_source = (ra_source**2. + dec_source**2.)**0.5

    #no need to consider multiple lensing if the source distance from axis is > einstein radius 
    #this has been checked empirically for the SIE+shear
    if dist_source>theta*1.05:
        #mult_imaged = False
        #print('too far', theta/dist_source)
        continue
    else:
        #print('maybe ok!', theta/dist_source)

        #draw an axis ratio according to OM10
        qval = 0.
        while ((qval>1) or (qval<0.1)):
            qval = np.random.normal(qmean, qstd)

        phi_e = np.random.uniform(0, np.pi)
        e1, e2 = param_util.phi_q2_ellipticity(phi=phi_e, q=qval)
        gamma = np.random.lognormal(mean=np.log(0.05), sigma=0.2*np.log(10))
        gamma_phi = np.random.uniform(0, 2.*np.pi)
        gamma1, gamma2 = param_util.shear_polar2cartesian(phi=gamma_phi, gamma=gamma)
        kwargs_shear = {'gamma1': gamma1, 'gamma2': gamma2, 'ra_0': 0, 'dec_0': 0}
        kwargs_spemd = {'theta_E': theta, 'center_x': 0, 'center_y': 0, 'e1': e1, 'e2': e2}
        kwargs_lens = [kwargs_spemd, kwargs_shear]
        lens_model_class = LensModel(lens_model_list=lens_model_list)
        lensEquationSolver = LensEquationSolver(lens_model_class)
        #ra_source, dec_source = 0.2, 0.4
        #t1 = time.time()
        x_image, y_image = lensEquationSolver.findBrightImage(ra_source, dec_source, kwargs_lens, min_distance=0.005, search_window=8., num_iter_max=20)
        mag = np.abs(lens_model_class.magnification(x_image, y_image, kwargs=kwargs_lens))
        
        if len(x_image)>1:

            #now magnitudes are important so draw the source mag
            imag, Msource =draw_source_givenz(z=zsource, imagmax=28, Mmin=-28, Mmax=-21, dM=0.1)

            imag_images = imag-2.5*np.log10(mag)
            
            if len(x_image)==2:
                if np.max(imag_images)>23.3:
                    print('imag2 failed')
                    continue
            if len(x_image)==4:
                if np.sort(imag_images)[2]>23.3:
                    print('imag3 failed')
                    continue

            print(imag_images, np.sort(imag_images))
            xs, ys = x_image, y_image
            maxdists = []
            for i in range(len(xs)):
                distances = ((xs[i]-xs)**2.+ (ys-ys[i])**2)**0.5
                maxdist = np.sort(distances)[-1]
                maxdists.append(maxdist)
            imsep = np.max(maxdists)
            if (imsep<0.5) or (imsep>4.0):
                print(np.max(maxdists), 'imsep failed')
                continue

            if len(x_image)==2:
                magratio = np.max(mag)/np.min(mag)
                if magratio>10:
                    thrown_out_doubles +=1
                    continue

            lens = {'x_source':ra_source, 
                    'y_source':dec_source,
                    'theta':theta,
                    'gamma':gamma,
                    'gamma_pa':gamma_phi,
                    'zlens':zlens,
                    'sigmalens':sigmalens,
                    'qlens':qval,
                    'phi_e':phi_e,
                    'imag':imag,
                    'zsource':zsource,
                    'Msource':Msource,
                    'ximg':x_image,
                    'yimg':y_image,
                    'magnification':mag,
                    'total_mag':mag.sum(),
                    'imsep':imsep}

            final_lenses.append(lens)
            i = nlenses//M
            j = nlenses - M*i
            print(nlenses, j/M, i/N)

            ax = plt.axes([j/M, i/N, 1/M, 1/N])
            print('found a lens', x_image, y_image, mag, qval, phi_e, gamma, gamma_phi)


            plot_lens(ax, x_image, y_image, phi_e, qval, gamma, gamma_phi, mag, ra_source, dec_source)
            nlenses += 1
            print(str(nlenses)+'/'+str(N*M), 'lenses;', ntries, 'tries')
            ntries = 0            

            #df
        else:
            continue
np.save(filename, np.array(final_lenses))
plt.show()
