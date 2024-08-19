# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 13:57:46 2024

@author: EDY
"""

import numpy as np
import pandas as pd
import pycwt as wavelet
import matplotlib.pylab as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
from scipy.special._ufuncs import gammainc, gamma
from scipy.optimize import fminbound
from Utils.data_processing import data_processing
from Utils.ordered_easydict import OrderedEasyDict as edict
from Module01.wrapped.func01_table_stats import table_stats
import os
import matplotlib

matplotlib.use('Agg')


def wavelet(Y, dt, pad=0, dj=-1, s0=-1, J1=-1, mother=-1, param=-1, freq=None):
    n1 = len(Y)

    if s0 == -1:
        s0 = 2 * dt
    if dj == -1:
        dj = 1. / 4.
    if J1 == -1:
        J1 = np.fix((np.log(n1 * dt / s0) / np.log(2)) / dj)
    if mother == -1:
        mother = 'MORLET'

    #....construct time series to analyze, pad if necessary
    x = Y - np.mean(Y)
    if pad == 1:
        base2 = np.fix(np.log(n1) / np.log(2) + 0.4999)  # power of 2 nearest to N
        x = np.concatenate((x, np.zeros((2**(base2 + 1) - n1).astype(np.int64))))

    n = len(x)

    #....construct wavenumber array used in transform [Eqn(5)]
    kplus = np.arange(1, int(n / 2) + 1)
    kplus = (kplus * 2 * np.pi / (n * dt))
    kminus = np.arange(1, int((n - 1) / 2) + 1)
    kminus = np.sort((-kminus * 2 * np.pi / (n * dt)))
    k = np.concatenate(([0.], kplus, kminus))

    #....compute FFT of the (padded) time series
    f = np.fft.fft(x)  # [Eqn(3)]

    #....construct SCALE array & empty PERIOD & WAVE arrays
    if mother.upper() == 'MORLET':
        if param == -1:
            param = 6.
        fourier_factor = 4 * np.pi / (param + np.sqrt(2 + param**2))
    elif mother.upper == 'PAUL':
        if param == -1:
            param = 4.
        fourier_factor = 4 * np.pi / (2 * param + 1)
    elif mother.upper == 'DOG':
        if param == -1:
            param = 2.
        fourier_factor = 2 * np.pi * np.sqrt(2. / (2 * param + 1))
    else:
        fourier_factor = np.nan

    if freq is None:
        j = np.arange(0, J1 + 1)
        scale = s0 * 2.**(j * dj)
        freq = 1. / (fourier_factor * scale)
        period = 1. / freq
    else:
        scale = 1. / (fourier_factor * freq)
        period = 1. / freq
    wave = np.zeros(shape=(len(scale), n), dtype=complex)  # define the wavelet array

    # loop through all scales and compute transform
    for a1 in range(0, len(scale)):
        daughter, fourier_factor, coi, _ = wave_bases(mother, k, scale[a1], param)
        wave[a1, :] = np.fft.ifft(f * daughter)  # wavelet transform[Eqn(4)]

    # COI [Sec.3g]
    coi = coi * dt * np.concatenate((np.insert(np.arange((n1 + 1) / 2 - 1), [0], [1E-5]), np.insert(np.flipud(np.arange(0, n1 / 2 - 1)), [-1], [1E-5])))
    wave = wave[:, :n1]  # get rid of padding before returning

    return wave, period, scale, coi


def wave_bases(mother, k, scale, param):
    n = len(k)
    kplus = np.array(k > 0., dtype=float)

    if mother == 'MORLET':  # -----------------------------------  Morlet

        if param == -1:
            param = 6.

        k0 = np.copy(param)
        expnt = -(scale * k - k0)**2 / 2. * kplus
        norm = np.sqrt(scale * k[1]) * (np.pi**(-0.25)) * np.sqrt(n)  # total energy=N   [Eqn(7)]
        daughter = norm * np.exp(expnt)
        daughter = daughter * kplus  # Heaviside step function
        fourier_factor = (4 * np.pi) / (k0 + np.sqrt(2 + k0**2))  # Scale-->Fourier [Sec.3h]
        coi = fourier_factor / np.sqrt(2)  # Cone-of-influence [Sec.3g]
        dofmin = 2  # Degrees of freedom
    elif mother == 'PAUL':  # --------------------------------  Paul
        if param == -1:
            param = 4.
        m = param
        expnt = -scale * k * kplus
        norm = np.sqrt(scale * k[1]) * (2**m / np.sqrt(m * np.prod(np.arange(1, (2 * m))))) * np.sqrt(n)
        daughter = norm * ((scale * k)**m) * np.exp(expnt) * kplus
        fourier_factor = 4 * np.pi / (2 * m + 1)
        coi = fourier_factor * np.sqrt(2)
        dofmin = 2
    elif mother == 'DOG':  # --------------------------------  DOG
        if param == -1:
            param = 2.
        m = param
        expnt = -(scale * k)**2 / 2.0
        norm = np.sqrt(scale * k[1] / gamma(m + 0.5)) * np.sqrt(n)
        daughter = -norm * (1j**m) * ((scale * k)**m) * np.exp(expnt)
        fourier_factor = 2 * np.pi * np.sqrt(2. / (2 * m + 1))
        coi = fourier_factor / np.sqrt(2)
        dofmin = 1
    else:
        print('Mother must be one of MORLET, PAUL, DOG')

    return daughter, fourier_factor, coi, dofmin


def wave_signif(Y, dt, scale, sigtest=0, lag1=0.0, siglvl=0.95, dof=None, mother='MORLET', param=None, gws=None):
    n1 = len(np.atleast_1d(Y))
    J1 = len(scale) - 1
    dj = np.log2(scale[1] / scale[0])

    if n1 == 1:
        variance = Y
    else:
        variance = np.std(Y)**2

    # get the appropriate parameters [see Table(2)]
    if mother == 'MORLET':  # ----------------------------------  Morlet
        empir = ([2., -1, -1, -1])
        if param is None:
            param = 6.
            empir[1:] = ([0.776, 2.32, 0.60])
        k0 = param
        fourier_factor = (4 * np.pi) / (k0 + np.sqrt(2 + k0**2))  # Scale-->Fourier [Sec.3h]
    elif mother == 'PAUL':
        empir = ([2, -1, -1, -1])
        if param is None:
            param = 4
            empir[1:] = ([1.132, 1.17, 1.5])
        m = param
        fourier_factor = (4 * np.pi) / (2 * m + 1)
    elif mother == 'DOG':  # -------------------------------------Paul
        empir = ([1., -1, -1, -1])
        if param is None:
            param = 2.
            empir[1:] = ([3.541, 1.43, 1.4])
        elif param == 6:  # --------------------------------------DOG
            empir[1:] = ([1.966, 1.37, 0.97])
        m = param
        fourier_factor = 2 * np.pi * np.sqrt(2. / (2 * m + 1))
    else:
        print('Mother must be one of MORLET, PAUL, DOG')

    period = scale * fourier_factor
    dofmin = empir[0]  # Degrees of freedom with no smoothing
    Cdelta = empir[1]  # reconstruction factor
    gamma_fac = empir[2]  # time-decorrelation factor
    dj0 = empir[3]  # scale-decorrelation factor

    freq = dt / period  # normalized frequency

    if gws is not None:  # use global-wavelet as background spectrum
        fft_theor = gws
    else:
        fft_theor = (1 - lag1**2) / (1 - 2 * lag1 * np.cos(freq * 2 * np.pi) + lag1**2)  # [Eqn(16)]
        fft_theor = variance * fft_theor  # include time-series variance

    signif = fft_theor
    if dof is None:
        dof = dofmin

    if sigtest == 0:  # no smoothing, DOF=dofmin [Sec.4]
        dof = dofmin
        chisquare = chisquare_inv(siglvl, dof) / dof
        signif = fft_theor * chisquare  # [Eqn(18)]
    elif sigtest == 1:  # time-averaged significance
        if len(np.atleast_1d(dof)) == 1:
            dof = np.zeros(J1) + dof
        dof[dof < 1] = 1
        dof = dofmin * np.sqrt(1 + (dof * dt / gamma_fac / scale)**2)  # [Eqn(23)]
        dof[dof < dofmin] = dofmin  # minimum DOF is dofmin
        for a1 in range(0, J1 + 1):
            chisquare = chisquare_inv(siglvl, dof[a1]) / dof[a1]
            signif[a1] = fft_theor[a1] * chisquare
    elif sigtest == 2:  # time-averaged significance
        if len(dof) != 2:
            print('ERROR: DOF must be set to [S1,S2], the range of scale-averages')
        if Cdelta == -1:
            print('ERROR: Cdelta & dj0 not defined for ' + mother + ' with param = ' + str(param))

        s1 = dof[0]
        s2 = dof[1]
        avg = np.logical_and(scale >= 2, scale < 8)  # scales between S1 & S2
        navg = np.sum(np.array(np.logical_and(scale >= 2, scale < 8), dtype=int))
        if navg == 0:
            print('ERROR: No valid scales between ' + str(s1) + ' and ' + str(s2))
        Savg = 1. / np.sum(1. / scale[avg])  # [Eqn(25)]
        Smid = np.exp((np.log(s1) + np.log(s2)) / 2.)  # power-of-two midpoint
        dof = (dofmin * navg * Savg / Smid) * np.sqrt(1 + (navg * dj / dj0)**2)  # [Eqn(28)]
        fft_theor = Savg * np.sum(fft_theor[avg] / scale[avg])  # [Eqn(27)]
        chisquare = chisquare_inv(siglvl, dof) / dof
        signif = (dj * dt / Cdelta / Savg) * fft_theor * chisquare  # [Eqn(26)]
    else:
        print('ERROR: sigtest must be either 0, 1, or 2')

    return signif


def chisquare_inv(P, V):

    if (1 - P) < 1E-4:
        print('P must be < 0.9999')

    if P == 0.95 and V == 2:  # this is a no-brainer
        X = 5.9915
        return X

    MINN = 0.01  # hopefully this is small enough
    MAXX = 1  # actually starts at 10 (see while loop below)
    X = 1
    TOLERANCE = 1E-4  # this should be accurate enough

    while (X + TOLERANCE) >= MAXX:  # should only need to loop thru once
        MAXX = MAXX * 10.
        # this calculates value for X, NORMALIZED by V
        X = fminbound(chisquare_solve, MINN, MAXX, args=(P, V), xtol=TOLERANCE)
        MINN = MAXX

    X = X * V  # put back in the goofy V factor

    return X  # end of code


def chisquare_solve(XGUESS, P, V):

    PGUESS = gammainc(V / 2, V * XGUESS / 2)  # incomplete Gamma function
    PDIFF = np.abs(PGUESS - P)  # error in calculated P
    TOL = 1E-4

    if PGUESS >= 1 - TOL:  # if P is very close to 1 (i.e. a bad guess)
        PDIFF = XGUESS  # then just assign some big number like XGUESS

    return PDIFF


def wavelet_main(df, output_filepath):
    df_new = df.copy()
    df_new['区域平均'] = df_new.iloc[:, :].mean(axis=1).round(1)
    df_new['区域最大'] = df_new.iloc[:, :].max(axis=1)
    df_new['区域最小'] = df_new.iloc[:, :].min(axis=1)
    
    columns = df_new.columns.tolist()
    year = df_new.index.tolist()
    year = [int(y) for y in year]

    all_result = edict()
    for i in range(len(columns)):
        col = columns[i]
        name = ''.join(col)
        dat = df_new.iloc[:, i].values

        if np.any(np.isnan(dat)):
            # print(f'{columns[i]}存在nan值，时间序列不完整')
            all_result[name] = '该站点的时间序列不完整，不能生成结果'
            continue

        if np.nanmax(dat)==0 and np.nanmin(dat)==0:
            # print(f'{columns[i]}存在nan值，时间序列不完整')
            all_result[name] = '该站点的数据全为0，不能生成结果'
            continue

        sst = dat
        # sst = sst - np.mean(sst)
        variance = np.std(sst, ddof=1)**2

        #----------C-O-M-P-U-T-A-T-I-O-N------S-T-A-R-T-S------H-E-R-E------------------------------------------------------
        # normalize by standard deviation (not necessary, but makes it easier
        # to compare with plot on Interactive Wavelet page, at
        # "http://paos.colorado.edu/research/wavelets/plot/"
        if 1:
            variance = 1.0
            sst = sst / np.std(sst, ddof=1)

        n = len(sst)
        dt = 1
        time = np.arange(len(sst)) * dt + year[0]  # construct time array
        xlim = ([year[0] - 1, year[-1] + 1])  # plotting range
        pad = 1  # pad the time series with zeroes (recommended)
        dj = 0.25  # this will do 4 sub-octaves per octave
        s0 = 0.5  # this says start at a scale of 6 months
        j1 = 7 / dj  # this says do 7 powers-of-two with dj sub-octaves each
        lag1 = 0.72  # lag-1 autocorrelation for red noise background
        # print("lag1 = ", lag1)
        mother = 'MORLET'

        # Wavelet transform:
        wave, period, scale, coi = wavelet(sst, dt, pad, dj, s0, j1, mother)
        power = (np.abs(wave))**2  # compute wavelet power spectrum
        global_ws = (np.sum(power, axis=1) / n)  # time-average over all times

        # Significance levels:
        signif = wave_signif(([variance]), dt=dt, sigtest=0, scale=scale, lag1=lag1, mother=mother)
        sig95 = signif[:, np.newaxis].dot(np.ones(n)[np.newaxis, :])  # expand signif --> (J+1)x(N) array
        sig95 = power / sig95  # where ratio > 1, power is significant

        # Global wavelet spectrum & significance levels:
        dof = n - scale  # the -scale corrects for padding at edges
        global_signif = wave_signif(variance, dt=dt, scale=scale, sigtest=1, lag1=lag1, dof=dof, mother=mother)

        # Scale-average between El Nino periods of 2--8 years
        avg = np.logical_and(scale >= 2, scale < 8)
        Cdelta = 0.776  # this is for the MORLET wavelet
        scale_avg = scale[:, np.newaxis].dot(np.ones(n)[np.newaxis, :])  # expand scale --> (J+1)x(N) array
        scale_avg = power / scale_avg  # [Eqn(24)]
        scale_avg = dj * dt / Cdelta * sum(scale_avg[avg, :])  # [Eqn(24)]
        scaleavg_signif = wave_signif(variance, dt=dt, scale=scale, sigtest=2, lag1=lag1, dof=([2, 7.9]), mother=mother)

        #------------------------------------------------------ Plotting

        #--- Plot time series
        fig = plt.figure(figsize=(9, 10))
        gs = GridSpec(3, 4, hspace=0.4, wspace=0.75)
        plt.subplots_adjust(left=0.1, bottom=0.05, right=0.9, top=0.95, wspace=0, hspace=0)
        plt.subplot(gs[0, 0:3])
        plt.plot(time, sst, 'k')
        plt.xlim(xlim[:])
        plt.xlabel('Time (year)')
        plt.ylabel('variance')
        plt.title('a) Time Series')

        #--- Contour plot wavelet power spectrum
        # plt3 = plt.subplot(3, 1, 2)
        plt3 = plt.subplot(gs[1, 0:3])
        levels = [0, 0.5, 1, 2, 4, 999]
        CS = plt.contourf(time, period, power, len(levels))  #*** or use 'contour'
        im = plt.contourf(CS, levels=levels, colors=['white', 'bisque', 'orange', 'orangered', 'darkred'])
        plt.xlabel('Time (year)')
        plt.ylabel('Period (years)')
        plt.title('b) Wavelet Power Spectrum')
        plt.xlim(xlim[:])
        # 95# significance contour, levels at -99 (fake) and 1 (95# signif)
        plt.contour(time, period, sig95, [-99, 1], colors='k')
        # cone-of-influence, anything "below" is dubious
        plt.plot(time, coi[:-1], 'k')
        # format y-scale
        plt3.set_yscale('log', base=2, subs=None)
        plt.ylim([np.min(period), np.max(period)])
        ax = plt.gca().yaxis
        ax.set_major_formatter(ticker.ScalarFormatter())
        plt3.ticklabel_format(axis='y', style='plain')
        plt3.invert_yaxis()
        # set up the size and location of the colorbar
        # position=fig.add_axes([0.5,0.36,0.2,0.01])
        # plt.colorbar(im, cax=position, orientation='horizontal') #, fraction=0.05, pad=0.5)

        # plt.subplots_adjust(right=0.7, top=0.9)

        #--- Plot global wavelet spectrum
        plt4 = plt.subplot(gs[1, -1])
        plt.plot(global_ws, period)
        plt.plot(global_signif, period, '--')
        plt.xlabel('Power (\u00B0C$^2$)')
        plt.title('c) Global Wavelet Spectrum')
        plt.xlim([0, 1.25 * np.max(global_ws)])
        # format y-scale
        plt4.set_yscale('log', base=2, subs=None)
        plt.ylim([np.min(period), np.max(period)])
        ax = plt.gca().yaxis
        ax.set_major_formatter(ticker.ScalarFormatter())
        plt4.ticklabel_format(axis='y', style='plain')
        plt4.invert_yaxis()

        # --- Plot 2--8 yr scale-average time series
        plt.subplot(gs[2, 0:3])
        plt.plot(time, scale_avg, 'k')
        plt.xlim(xlim[:])
        plt.xlabel('Time (year)')
        plt.ylabel('Avg variance')
        plt.title('d) 2-8 yr Scale-average Time Series')
        plt.plot(xlim, scaleavg_signif + [0, 0], '--')

        result_picture = os.path.join(output_filepath, name+'_小波.png')
        fig.savefig(result_picture, dpi=200, bbox_inches='tight')
        plt.clf()
        plt.close()

        all_result[name] = result_picture

    return all_result


if __name__ == "__main__":
    path = r'C:/Users/MJY/Desktop/qhkxxlz/app/Files/test_data/qh_mon.csv'
    df = pd.read_csv(path, low_memory=False)
    element = 'TEM_Avg'
    df = df[['Station_Id_C', 'Station_Name', 'Lat', 'Lon', 'Datetime', 'Year', 'Mon', element]]
    df = data_processing(df, element)
    data_df = df[df.index.year <= 2011]
    refer_df = df[(df.index.year > 2000) & (df.index.year < 2020)]
    nearly_df = df[df.index.year > 2011]
    last_year = 2023
    stats_result, post_data_df, post_refer_df = table_stats(data_df, refer_df, nearly_df, element, last_year)

    save_file = r'C:/Users/MJY/Desktop/result'
    all_result = wavelet_main(post_data_df, save_file)
