import numpy as np
import qutip
from qutip import *
from qutip.solver import correlation_2op_1t
import pandas as pd
from joblib import Parallel, delayed

#ANALYTICAL CALCULATIONS
def Transition_rates(g0, alpha2, kappa, Delta, wm):
    """""""""
    Calculates the transition rates of the molecular optomechanical model
    g0: Coupling rate
    alpha2: coherent state defining the cavity field (squared)
    kappa: cavity field's total decay rate
    wm: Frequency of the molecular/mechanical mode
    """""""""
    Delta_prima=Delta-2*g0*alpha2/wm
    #Transition rates
    Gamma_plus=g0**2*alpha2*kappa/((Delta+wm)**2+(kappa/2)**2) #Creation of phonon
    Gamma_minus=g0**2*alpha2*kappa/((Delta-wm)**2+(kappa/2)**2) #Annihilation of phonon
    #Optomechanical damping rate
    Gamma_opt=Gamma_minus-Gamma_plus
    return Gamma_plus, Gamma_minus, Gamma_opt

def pop_phonons(gamma_m, g_opt, n_th, g_plus):
    """""""""
    Calculates the phonon pupulation of the system (population of the mechanical mode)
    gamma_m: Mechanical mode's total decay rate
    g_opt: Optomechanical damping
    nth: Thermal population
    g_plus: Transition rate of creation of phonons
    """""""""
    #Population of phonons
    n=gamma_m/(gamma_m+g_opt)*n_th+g_plus/(gamma_m+g_opt)
    return n

def emission_spec(w, nb, g_plus, g_minus, g_opt, gamma_m, wl, wm):
    """""""""
    Calculates the emission spectrum analytically
    w: Frequency at which the spectrum is calculated
    nb: Phonon population
    g_plus: Transition rate of creation of phonons
    g_minus: Transition rate of annihilation of phonons
    g_opt: Optomechanical damping
    gamma_m: Mechanical mode's total decay rate
    wl: Laser's frequency
    wm: Frequency of the molecular/mechanical mode
    """""""""
    s1=nb*w**4*g_minus*((g_opt+gamma_m)/2)/((w-wl-wm)**2+((g_opt+gamma_m)**2/4))
    s2= (nb+1)*w**4*g_plus*((g_opt+gamma_m)/2)/((w-wl+wm)**2+((g_opt+gamma_m)**2/4))
    S = s1 + s2
    return S
    
def S_ws(wl, wm, g_plus, n_b, g_opt, gamma_m):
    """""""""
    Calculates the stokes intensity analytically
    wl: Laser's frequency
    wm: Frequency of the molecular/mechanical mode
    g_plus: Transition rate of creation of phonons
    n_b: Phonon population
    gamma_m: Mechanical mode's total decay rate
    """""""""
    ws=wl-wm
    S=ws**4*g_plus*(n_b+1)/(g_opt+gamma_m)
    return S

def S_was(wl, wm, g_minus, n_b, g_opt, gamma_m):
    """""""""
    Calculates the anti-stokes intensity analitically
    wl: Laser's frequency
    wm: Frequency of the molecular/mechanical mode
    g_minus: Transition rate of annihilation of phonons
    n_b: Phonon population
    gamma_m: Mechanical mode's total decay rate
    """""""""
    was=wl+wm
    S=was**4*g_minus*n_b/(g_opt+gamma_m)
    return S

#NUMERICAL CALCULATIONS
def raman_signals(s_order, as_order, w, wl, wm, S):
    """""""""
    Calculates numerically the Raman signals: the frequencies of the raman processes, the corresponding intensities and the 
    linewidth of the emission signals.
    
    s_order: Order of the Stokes process sideband
    as_order: Order of the anti-Stokes process sideband
    w: Frequency at which the spectrum is calculated
    wl: Laser's frequency
    wm: Frequency of the molecular/mechanical mode
    S: Emission spectrum
    """""""""
    indx_stokes = np.abs(w- (wl-s_order*wm)).argmin() #Index of stokes
    indx_antistokes = np.abs(w- (wl+as_order*wm)).argmin() #Index of antistokes
    w_stokes = np.real(w[indx_stokes]) #stokes frequency
    w_antistokes = np.real(w[indx_antistokes]) #antistokes frequency
    S_stokes = np.real(S[indx_stokes]) #stokes intensity
    S_antistokes = np.real(S[indx_antistokes]) #antistokes intensity
    w_range=0.1 
    indx1_s=np.abs(w-(w_stokes-w_range)).argmin()
    indx2_s=np.abs(w-(w_stokes+w_range)).argmin()
    indx1_as=np.abs(w-(w_antistokes-w_range)).argmin()
    indx2_as=np.abs(w-(w_antistokes+w_range)).argmin()
    w_s=np.real(w[indx1_s:indx2_s]) #stokes band w
    S_s=np.real(S[indx1_s:indx2_s]) #stokes band S
    w_as=np.real(w[indx1_as:indx2_as]) #antistokes band w
    S_as=np.real(S[indx1_as:indx2_as]) #antistokes band S
    indx_stokes = np.abs(w_s- (wl-s_order*wm)).argmin() #Index of stokes redefined
    indx_antistokes = np.abs(w_as- (wl+as_order*wm)).argmin() #Index of antistokes redefined
    half_max_stokes=S_stokes/2
    half_max_antistokes=S_antistokes/2
    indx_half_max_stokes = np.abs(S_s- half_max_stokes).argmin()
    indx_half_max_antistokes = np.abs(S_as- half_max_antistokes).argmin()
    if indx_half_max_stokes > indx_stokes:
        half_spec=S_s[0:indx_stokes]
        half_w=w_s[0:indx_stokes]
        indx_half_max_stokes1 = np.abs(half_spec - half_max_stokes).argmin()
        width_stokes=abs(w_s[indx_half_max_stokes]-half_w[indx_half_max_stokes1])
    elif indx_half_max_stokes < indx_stokes:
        last_indx=len(S_s)-1
        half_spec=S_s[indx_stokes:last_indx]
        half_w=w_s[indx_stokes:last_indx]
        indx_half_max_stokes1 = np.abs(half_spec - half_max_stokes).argmin()
        width_stokes=abs(half_w[indx_half_max_stokes1]-w_s[indx_half_max_stokes])
    else:
        width_stokes=0
        
    if indx_half_max_antistokes > indx_antistokes:
        half_spec=S_as[0:indx_antistokes]
        half_w=w_as[0:indx_antistokes]
        indx_half_max_antistokes1 = np.abs(half_spec - half_max_antistokes).argmin()
        width_antistokes=abs(w_as[indx_half_max_antistokes]-half_w[indx_half_max_antistokes1])
    elif indx_half_max_antistokes < indx_antistokes:
        last_indx=len(S_as)-1
        half_spec=S_as[indx_antistokes:last_indx]
        half_w=w_as[indx_antistokes:last_indx]
        indx_half_max_antistokes1 = np.abs(half_spec - half_max_antistokes).argmin()
        width_antistokes=abs(half_w[indx_half_max_antistokes1]-w_as[indx_half_max_antistokes])
    else:
        width_antistokes=0
        
    stokes=[w_stokes, S_stokes, width_stokes]
    antistokes=[w_antistokes, S_antistokes, width_antistokes]
    return stokes, antistokes

def phonon_pop_from_S(s_as, s_s, g_plus, g_minus, wl, wm):
    """""""""
    Calculates the phonon population using the Raman signals
    s_as: anti-stokes intensity
    s_s: stokes intensity
    g_plus: Transition rate of creation of phonons
    g_minus: Transition rate of annihilation of phonons
    wl: Laser's frequency
    wm: Frequency of the molecular/mechanical mode
    """""""""
    ws=wl-wm
    was=wl+wm
    rateS=s_s/s_as
    rateW=was/ws
    rateG=g_minus/g_plus
    n=1/((rateS*rateW**4*rateG)-1)
    return n

def nb_for_Delta_0(nth, Omega, g0, kappa, wm, gamma_m):
    """""""""
    Calculates the phonon population for nule detuning
    nth: Thermal population
    Omega: Laser drive
    g0: Coupling rate
    kappa: cavity field's total decay rate
    wm: Frequency of the molecular/mechanical mode
    gamma_m: Mechanical mode's total decay rate
    """""""""
    s2=(4*g0/(kappa*abs(kappa-2*1j*wm)))**2
    nb=nth+s2*Omega**2*kappa/gamma_m
    return nb

def nth_from_nb(nb, gamma_m, g_opt, g_plus):
    """""""""
    Calculates the thermal population using the phonon population
    nb: phonon population
    gamma_m: Mechanical mode's total decay rate
    g_opt: Optomechanical damping
    g_plus: Transition rate of creation of phonons
    """""""""
    return (gamma_m+g_opt)*nb/gamma_m-g_plus/gamma_m


def number_of_points(w, wl, wm, w_range=0.004):
    """""""""
    Calculates the number of points of a given peak
    w: Frequency at which the spectrum is calculated
    wl: Laser's frequency
    wm: Frequency of the molecular/mechanical mode
    w_range: The width of the peak considered when calculating the points in it. Default=0.004.
    """""""""
    indx_stokes = np.abs(w- (wl-wm)).argmin()
    indx_antistokes = np.abs(w- (wl+wm)).argmin()
    w_s=w[indx_stokes]
    w_as=w[indx_antistokes]
    indx1_s=np.abs(w-(w_s-w_range)).argmin()
    indx2_s=np.abs(w-(w_s+w_range)).argmin()
    indx1_as=np.abs(w-(w_as-w_range)).argmin()
    indx2_as=np.abs(w-(w_as+w_range)).argmin()
    points_s=len(w[indx1_s:indx2_s])
    points_as=len(w[indx1_as:indx2_as])
    return points_s, points_as

def calculate_noisefloor(w, wl, wm, S):
    """""""""
    Calculates the baseline of the emission spectrum
    w: Frequency at which the spectrum is calculated
    wl: Laser's frequency
    wm: Frequency of the molecular/mechanical mode
    S: Emission spectrum
    """""""""
    range_w=0.05
    indx_stokes1 = np.abs(w- (wl-3*wm)).argmin() #Index of stokes
    indx_antistokes1 = np.abs(w- (wl+3*wm)).argmin() #Index of antistokes
    indx_stokes2 = np.abs(w- (wl-4*wm)).argmin() #Index of stokes
    indx_antistokes2 = np.abs(w- (wl+4*wm)).argmin()
    right_s1=w[indx_stokes1]+range_w
    left_s1=w[indx_stokes1]-range_w
    right_as1=w[indx_antistokes1]+range_w
    left_as1=w[indx_antistokes1]-range_w
    right_s2=w[indx_stokes2]+range_w
    left_s2=w[indx_stokes2]-range_w
    right_as2=w[indx_antistokes2]+range_w
    left_as2=w[indx_antistokes2]-range_w
    indx_noise_s_r1= np.abs(w- right_s1).argmin()
    indx_noise_s_l1= np.abs(w- left_s1).argmin()
    indx_noise_as_r1= np.abs(w- right_as1).argmin()
    indx_noise_as_l1= np.abs(w- left_as1).argmin()
    indx_noise_s_r2= np.abs(w- right_s2).argmin()
    indx_noise_s_l2= np.abs(w- left_s2).argmin()
    indx_noise_as_r2= np.abs(w- right_as2).argmin()
    indx_noise_as_l2= np.abs(w- left_as2).argmin()
    av_noise= np.mean([S[indx_noise_s_r1], S[indx_noise_s_l1], S[indx_noise_as_r1], S[indx_noise_as_l1], S[indx_noise_s_r2], S[indx_noise_s_l2], S[indx_noise_as_r2], S[indx_noise_as_l2]])
    return av_noise, [indx_noise_s_r1, indx_noise_s_l1, indx_noise_as_r1, indx_noise_as_l1, indx_noise_s_r2, indx_noise_s_l2, indx_noise_as_r2, indx_noise_as_l2]

def spectrum_calculation(H, tlist, wl, wm, c_ops, a, b):
    """""""""
    Calculates the quantities of iterest for a given Hamiltonian. In this case, the emission spectrum (S), the frequency range (w), 
    the Raman intensities (S_stokes, S_antistokes) and frequencies (W_stokes, W_antistokes) and 
    the linewidth of the Raman signals (FWHM_stokes, FWHM_antistokes).
    H: Hamiltonian
    """""""""
    rho_ss = steadystate(H, c_ops, method="power", solver="spsolve")
    nb = qutip.expect(b.dag()*b, rho_ss)
    na = qutip.expect(a.dag()*a, rho_ss)
    corr = qutip.correlation_2op_1t(H, rho_ss, tlist, c_ops, a.dag(), a)
    wlist, spec = qutip.spectrum_correlation_fft(tlist, corr)
    w_shifted = wlist + wl
    s1, as1 = raman_signals(1, 1, np.array(w_shifted), wl, wm, spec)
    s2, as2 = raman_signals(2, 2, np.array(w_shifted), wl, wm, spec)
    s3, _ = raman_signals(3, 2, np.array(w_shifted), wl, wm, spec)
    noise_val, noise_p=calculate_noisefloor(np.array(w_shifted), wl, wm, spec)
    noise_w_p=[w_shifted[noise_p[0]], w_shifted[noise_p[1]], w_shifted[noise_p[2]], w_shifted[noise_p[3]], w_shifted[noise_p[4]], w_shifted[noise_p[5]], w_shifted[noise_p[6]], w_shifted[noise_p[7]]]
    noise_s_p=[spec[noise_p[0]], spec[noise_p[1]],spec[noise_p[2]], spec[noise_p[3]], spec[noise_p[4]], spec[noise_p[5]],spec[noise_p[6]], spec[noise_p[7]]]
    return {
        'S': spec,
        'w': w_shifted,
        'Noise Floor': noise_val,
        'baseline_w_p': noise_w_p,
        'baseline_s_p': noise_s_p,
        'na': na,
        'nb': nb,
        'S1': np.real(s1[1]),
        'S2': np.real(s2[1]),
        'S3': np.real(s3[1]),
        'AS1': np.real(as1[1]),
        'AS2': np.real(as2[1]),
        'wS1': np.real(s1[0]),
        'wS2': np.real(s2[0]),
        'wS3': np.real(s3[0]),
        'wAS1': np.real(as1[0]),
        'wAS2': np.real(as2[0]),
        'FWHM_S1': s1[2],
        'FWHM_S2': s2[2],
        'FWHM_S3': s3[2],
        'FWHM_AS1': as1[2],
        'FWHM_AS2': as2[2]}


