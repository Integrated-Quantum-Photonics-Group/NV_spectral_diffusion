MIT License

Copyright (c) [year] [fullname]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import sympy as sp
import pickle as pckl
from scipy.special import wofz
from numpy import sin, cos
from scipy.special import i0, i1, k0, k1, iv, kv
import scipy.integrate as integrate
from numba import jit

#constants
e_charge = sc.constants.elementary_charge
epsilon_r = 5.5
epsilon_0 = sc.constants.epsilon_0

zvec = np.array([0,0,1])

#for the cyl. symmetric el. field
epsilon1 = epsilon_r
epsilon2 = 1

#conversion factors
pi = np.pi
Nano = 10**(-9)
Peta = 10**(15)
Tera = 10**(12)
Giga = 10**(9)
Mega = 10**6
Micro = 10**(-6)
Milli = 10**(-3)

# NV- energies
eA20 = 0
eA2m = 2.87 * Milli
eA2p = 2.87 * Milli
eE1 = 470.2989 - 5.73 * Milli
eE2 = 470.2989 - 5.73 * Milli
eEx = 470.2989 - 0.65 * Milli
eEy = 470.2989 - 0.65 * Milli
eA1 = 470.2989 - 3.77 * Milli
eA2 = 470.2989 - 5.73 * Milli

# electric field coupling constants
g = 2 * Peta
a = b = c = 0.3 * Micro / Mega
d = 3 * Micro / Mega

# elecrtic field prefactor
el_fac = e_charge / (4 * pi * epsilon_0 * epsilon_r * Nano**2)

# nano pillar dimensions (flat top)

pi = np.pi
z_Max = 1600
r_Max = 125
R0 = r_Max
w0_FWHM = 13 * Milli

theta = 2 * pi * 35/360

rot_x = np.array([
    [1, 0 , 0],
    [0, np.cos(theta), -np.sin(theta)],
    [0, np.sin(theta), np.cos(theta)]
]
)

def lorentz(omega,omega0):
    x = 2 * (omega - omega0)/w0_FWHM
    return 1/(1 + x ** 2)

def lorentz_sciup(omega,omega0):
    x = 2 * (omega - omega0)/.5
    return 1/(1 + x ** 2)

def lorentzfit(omega,omega0,w0_FWHM):
    x = 2 * (omega - omega0)/w0_FWHM
    return 1/(1 + x ** 2)

def gauss(omega,omega0):
    return np.exp(-(omega - omega0)**2 / 2/ w0_FWHM**2) / np.sqrt(2 * pi )/ w0_FWHM

def gaussfit(omega,amp, omega0, w0_FWHM):
    return amp * np.exp(-(omega - omega0)**2 / 2/ w0_FWHM**2)

def inhomline_gauss(omega, points):
    return sum([gauss(omega, i) for i in points])

def inhomline(omega, points):
    return sum([lorentz(omega, i) for i in points])

def electric_field_rot(pos_n, charge, **kwargs):
    ef = kwargs.get('elfac', el_fac)
    return np.dot(rot_x, sum([charge[k] * ef * np.array([pos[0], pos[1], pos[2]]) / (pos[0]**2 + pos[1]**2 + pos[2]**2)**(3/2) for k, pos in enumerate(pos_n)]))

def electric_field(pos_n, charge, **kwargs):
    ef = kwargs.get('elfac', el_fac)
    return sum([charge[k] * ef * np.array([pos[0], pos[1], pos[2]]) / (pos[0]**2 + pos[1]**2 + pos[2]**2)**(3/2) for k, pos in enumerate(pos_n)])

def energies(pos_n,charge):
    e_f = electric_field(pos_n, charge)
    e_arr = np.array([eA20 * Tera + 2 * b * g * e_f[2],
              (eEx * Tera + eEy * Tera - np.sqrt(eEx ** 2 * Tera ** 2 - 2 * eEx * eEy * Tera ** 2 +
                                                 eEy ** 2 * Tera ** 2 + 4 * a * eEx * g * Tera * e_f[0] -
                                                 4 * a * eEy * g * Tera * e_f[0] + 4 * a ** 2 * g ** 2 * e_f[0] ** 2 +
                                                 4 * a ** 2 * g ** 2 * e_f[1] ** 2)  + 2 * b * g * e_f[2] + 2 * d * g * e_f[2] ) / 2,
              (eEx * Tera + eEy * Tera + np.sqrt(eEx ** 2 * Tera ** 2 - 2 * eEx * eEy * Tera ** 2 +
                                                 eEy ** 2 * Tera ** 2 + 4 * a * eEx * g * Tera * e_f[0] -
                                                 4 * a * eEy * g * Tera * e_f[0] + 4 * a ** 2 * g ** 2 * e_f[0] ** 2 +
                                                 4 * a ** 2 * g ** 2 * e_f[1] ** 2) + 2 * b * g * e_f[2] + 2 * d * g * e_f[2]) / 2 ]) / Tera  # energies in THz, expressions taken from Mathematica
    return  e_arr


def diff(pos_n, charge) :
    ens = energies(pos_n, charge)
    a1 = (eEy - (ens[1] - ens[0])) * 1000 # *1000 = conversion to GHz
    a2 = (eEy - (ens[2] - ens[0])) * 1000 # *1000 = conversion to GHz
    return a1, a2

def energies_rot(pos_n,charge):
    e_f = electric_field_rot(pos_n, charge)
    e_arr = np.array([eA20 * Tera + 2 * b * g * e_f[2],
              (eEx * Tera + eEy * Tera - np.sqrt(eEx ** 2 * Tera ** 2 - 2 * eEx * eEy * Tera ** 2 +
                                                 eEy ** 2 * Tera ** 2 + 4 * a * eEx * g * Tera * e_f[0] -
                                                 4 * a * eEy * g * Tera * e_f[0] + 4 * a ** 2 * g ** 2 * e_f[0] ** 2 +
                                                 4 * a ** 2 * g ** 2 * e_f[1] ** 2)  + 2 * b * g * e_f[2] + 2 * d * g * e_f[2] ) / 2,
              (eEx * Tera + eEy * Tera + np.sqrt(eEx ** 2 * Tera ** 2 - 2 * eEx * eEy * Tera ** 2 +
                                                 eEy ** 2 * Tera ** 2 + 4 * a * eEx * g * Tera * e_f[0] -
                                                 4 * a * eEy * g * Tera * e_f[0] + 4 * a ** 2 * g ** 2 * e_f[0] ** 2 +
                                                 4 * a ** 2 * g ** 2 * e_f[1] ** 2) + 2 * b * g * e_f[2] + 2 * d * g * e_f[2]) / 2 ]) / Tera  
    return  e_arr

def diff_rot(pos_n, charge) :
    ens = energies_rot(pos_n, charge)
    a1 = (eEy - (ens[1] - ens[0])) * 1000
    a2 = (eEy - (ens[2] - ens[0])) * 1000
    return a1, a2

def V(x, x0, amp, alpha, gamma):
    """
    Return the Voigt line shape at x with Lorentzian component HWHM gamma
    and Gaussian component HWHM alpha.
    """
    sigma = alpha / np.sqrt(2 * np.log(2))
    return amp * np.real(wofz((x-x0 + 1j*gamma)/sigma/np.sqrt(2))) / sigma\
                                                           /np.sqrt(2*np.pi)

def rhovec(phi):
    return np.array([cos(phi),sin(phi),0])

def kernelz(k, z, rho):
    return -k * sin(k * z) *  ( (1 - epsilon1/epsilon2)/ (1 +  (k0(k * R0) * i1(k* R0)/k1(k*R0)/i0(k*R0)) * epsilon1/epsilon2)) * k0(k * R0) * i0(k * rho) / i0(k * R0)

def kernelrho(k, z, rho):
    return k * cos(k * z) * i1(k * rho) * k1(k * R0)  * ( (1 - epsilon1/epsilon2)/ (1 +  (k1(k * R0) * (i0(k* R0) + iv(2,k*R0) )/(k0(k*R0)+ kv(2, k * R0))/i1(k*R0)) * epsilon1/epsilon2)) / i1(k * R0)

def zfield(rho, z):
    return el_fac * integrate.quad(kernelz, 0, .08, args=(z, rho), limit=200)[0] * 2 / pi

def rhofield(rho, z):
    return el_fac * integrate.quad(kernelrho, 0, .08, args=(z, rho),limit=200)[0] * 2 / pi

def phi_rot(phi):
    return np.array([[cos(phi),-sin(phi),0],[sin(phi), cos(phi),0],[0,0,1]])

def pol_field(pos):
    x = pos[0]
    y = pos[1]
    z = pos[2]
    rho = np.sqrt(x**2 + y**2)
    if rho > 0:
        rho_vec = np.array([x / rho, y / rho, 0])
    else :
        rho_vec = np.array([0,0,0])
    rho_fac = rhofield(rho,z)
    z_fac = zfield(rho, z)
    out_field = z_fac * zvec + rho_fac * rho_vec
    return out_field

def el_field(pos):
    return el_fac * np.array([pos[0], pos[1], pos[2]]) / (pos[0]**2 + pos[1]**2 + pos[2]**2)**(3/2)

def cyl_field(pos):
    return (el_field(pos) - pol_field(pos))

def cyl_field_rot(pos):
    return np.dot(rot_x, (el_field(pos) - pol_field(pos)))

def energies_cyl_rot(field):
    e_f = field
    e_arr = np.array([eA20 * Tera + 2 * b * g * e_f[2],
              (eEx * Tera + eEy * Tera - np.sqrt(eEx ** 2 * Tera ** 2 - 2 * eEx * eEy * Tera ** 2 +
                                                 eEy ** 2 * Tera ** 2 + 4 * a * eEx * g * Tera * e_f[0] -
                                                 4 * a * eEy * g * Tera * e_f[0] + 4 * a ** 2 * g ** 2 * e_f[0] ** 2 +
                                                 4 * a ** 2 * g ** 2 * e_f[1] ** 2)  + 2 * b * g * e_f[2] + 2 * d * g * e_f[2] ) / 2,
              (eEx * Tera + eEy * Tera + np.sqrt(eEx ** 2 * Tera ** 2 - 2 * eEx * eEy * Tera ** 2 +
                                                 eEy ** 2 * Tera ** 2 + 4 * a * eEx * g * Tera * e_f[0] -
                                                 4 * a * eEy * g * Tera * e_f[0] + 4 * a ** 2 * g ** 2 * e_f[0] ** 2 +
                                                 4 * a ** 2 * g ** 2 * e_f[1] ** 2) + 2 * b * g * e_f[2] + 2 * d * g * e_f[2]) / 2 ]) / Tera  
    return  e_arr

def diff_cyl_rot(field) :
    ens = energies_cyl_rot(field)
    a1 = (eEy - (ens[1] - ens[0])) * 1000 # conversion to GHz
    a2 = (eEy - (ens[2] - ens[0])) * 1000 # conversion to GHz
    return a1, a2

def main():
    pass

if __name__ == '__main__':
    main()
