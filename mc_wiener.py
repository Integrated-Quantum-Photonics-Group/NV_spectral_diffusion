from numba import  njit
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import wofz

# global constants
Milli = 10**(-3)
Nano = 10**(-9)
Mega = 10**6
Giga = 10**9
pi = np.pi

# problem specific constants
w0_FWHM = 13 * Milli
eta = 3.4169 * 10**24 #sample specific quantity


def V(x, x0, amp, alpha, gamma):
    """
    Return the Voigt line shape at x with Lorentzian component HWHM gamma
    and Gaussian component HWHM alpha.
    """
    sigma = alpha / np.sqrt(2 * np.log(2))
    return amp * np.real(wofz((x-x0 + 1j*gamma)/sigma/np.sqrt(2))) / sigma\
                                                           /np.sqrt(2*np.pi)

@njit
def lorentzjit(omega,omega0):
    x = 2 * (omega - omega0)/w0_FWHM
    return 1/(1 + x ** 2)


@njit
def inhomlinejit(omega, points):
    out_arr = np.zeros(len(omega))
    k = 0
    for om in omega:
        s = 0
        for i in points:
            s = s + lorentzjit(om, i)
        out_arr[k] = s
        k = k + 1
    return out_arr


def gauss(x, sigma, a, x0):
    return a * np.exp(- (x-x0)**2/ 2 / sigma**2)

def pow_law(x,x0,a,b):
    return a * (x-x0)**b

def sdrs_func(intensity):
    return np.sqrt(2 * intensity * Nano * eta / pi / dt) / Mega

# function that returns a list of intensities and the corresponding anticipated inhomegeneously broadened linewidths
def simulation(**kwargs):


    plt_result = kwargs.get('plt_result', True)

    #time step in seconds
    dt = 2.3

    n_time_steps = 100
    
    intensity_range = np.linspace(.01,10,101)
    
    number_of_trajectories = 500
    
    FWHMS = []

    for I in intensity_range:

        y = np.zeros(n_time_steps)
        t = dt * np.arange(n_time_steps)

        frequs = []
        for i in range(number_of_trajectories):

            # wiener process modelleing diffusion
            y[-1] = Mega *  np.random.normal(loc=0.0, scale=13.0) # initial condition
            noise = np.random.normal(loc=0.0, scale=1.0, size=n_time_steps) * np.sqrt(dt)  # define noise process

            for i in range(0,n_time_steps):
                y[i] = y[i-1] +  np.sqrt(I * Nano * eta) * noise[i]
                frequs.append(y[i])

        frequs = np.array(frequs)/ Giga
        omega = np.linspace(-5,5,1001)

        # create inhomogeneous line from diffusion
        line = inhomlinejit(omega, frequs)
        line = line/max(line)

        # fit resulting line with a Voigt profile
        init_vals = [0, 1, .1, .1]  # for [x0, amp, alpha, gamma]
        best_vals, covar = curve_fit(V, omega, line, p0=init_vals, maxfev=10000)

        lwl = 2 * best_vals[3]
        lwg = 2 * 2 * best_vals[2]

        #extract full width half maximum from voigt profile
        FWHM = 0.5346 * lwl + np.sqrt(0.2166 * lwl ** 2 + lwg ** 2)
        FWHMS.append(FWHM)

    if plt_result:
        plt.plot(intensity_range, FWHMS, linestyle = 'none', marker = '.')
        plt.xlabel('$\\rm I/nW$')
        plt.ylabel('$\\rm FWHM/GHz$')
        plt.tight_layout()
        plt.show()
        return intensity_range, FWHMS
    else:
        return intensity_range, FWHMS

def main():
    simulation()
    return 0

if __name__ == '__main__':
    main()
