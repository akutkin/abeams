#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Try to restore primary beam shape by comparison source fluxes with NVSS

Created on Sat Jun 13 16:26:13 2020

@author: kutkin
"""
import matplotlib
matplotlib.use('Agg')


import os
import sys
import glob
import numpy as np
from shutil import copyfile
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import Angle
from astropy.coordinates import SkyCoord
from astropy.convolution import convolve
from astropy.visualization import ZScaleInterval, ImageNormalize
from astropy.wcs import WCS
from radio_beam import Beam, Beams


from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, RationalQuadratic, Matern
from sklearn.gaussian_process.kernels import ConstantKernel
from sklearn import preprocessing
from scipy import interpolate
from itertools import product

import matplotlib.pyplot as plt

import pandas as pd
import datetime

import logging

import pickle as pkl

import bdsf

if not '/home/kutkin/apertif' in sys.path:
    sys.path.append('/home/kutkin/apertif')

from atools.utils import fits_reconvolve_psf, get_tid_beam, fits_operation
from atools.catalogs import cross_match, load_nvss
from dataqa.continuum.validation_tool import (radio_image, validation)


nvss_psf = Beam(45*u.arcsec, 45*u.arcsec, 0*u.deg)

def process(beam):
    beam = str(beam).zfill(2)
    if not os.path.exists(f'/kutkin/beams/{beam}'):
        os.mkdir(f'/kutkin/beams/{beam}')
    files = glob.glob('/kutkin/acat/dr1/*_{}.fits'.format(beam))
    logging.info('%d files for beam %s', len(files), beam)
    df2 = load_nvss()
    x = []
    y = []
    z = []

    for f in files:
        tid, beam = get_tid_beam(f)
        basename = os.path.basename(f).rstrip('.fits')
        cat1 = f'/kutkin/beams/{beam}/' + basename+'.csv'
        hdrfile = f'/kutkin/beams/{beam}/' + basename+'.hdr'
        commoncat = f'/kutkin/beams/{beam}/' + basename+'_cm.csv'
        if os.path.exists(cat1) and os.path.exists(commoncat) and os.path.exists(hdrfile):
            logging.debug('Files exist. Skipping...')
            continue
        tmp = f'/kutkin/beams/{beam}/{tid}_{beam}_tmp.fits'
        copyfile(f, tmp)
# reconvolve with NVSS PSF:
        tmp=fits_reconvolve_psf(tmp, nvss_psf)
# source finding
        img = bdsf.process_image(tmp)
        img.write_catalog(outfile=cat1, format='csv', clobber=True, catalog_type='srl')
# save header
        hdr = fits.getheader(tmp)
        wcs = WCS(hdr).dropaxis(-1).dropaxis(-1)
        df = cross_match(cat1, df2)
        df.to_csv(commoncat, index=False)
        radec = [list(df.RA_1.values), list(df.DEC_1.values)]
        xi, yi = wcs.all_world2pix(radec[0], radec[1], 1)
        xi = [_-1537 for _ in xi]
        yi = [_-1537 for _ in yi]
        zi = list(df.Total_flux_1.values/df.Total_flux_2.values)
        x += xi
        y += yi
        z += zi
        with open(f'/kutkin/beams/{beam}/result.dat', 'a+') as out:
            for i,j,k in zip(xi,yi,zi):
                out.write(f'{tid} {beam} {i} {j} {k}\n')
        # print(df)
        hdr.tofile(hdrfile)
        os.remove(tmp)
    return


def gpr(x,y,z):
    """
    note the narrow priors -- l=340 ~ 22 arcmin for apercal (!) images
    """
    zmean = np.mean(z)
    kernel = ConstantKernel(0.2) * RBF([330, 330], (280, 400)) + \
              ConstantKernel(0.05) * RBF([0.1, 0.1], (1e-2, 1))
             # ConstantKernel(0.05) * Matern([.1,.1], (1e-3,10), 1.5)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=25)
    logging.info(gp.kernel)
    X = np.stack((x,y)).T
    gp.fit(X, z-zmean)
    logging.info(gp.kernel_)
    return gp


def gpsave(gp, fname):
    with open(fname, 'wb') as inp:
        pkl.dump(gp, inp)


def gpload(fname):
    with open(fname, 'rb') as out:
        return pkl.load(out)


def gppredict(gp, shape=1200, size=100):
    x1 = np.linspace(-shape, shape, size) #p
    x2 = np.linspace(-shape, shape, size) #q
    x1x2 = np.array(list(product(x1, x2)))
    y_pred, MSE = gp.predict(x1x2, return_std=True)
    Zp = np.reshape(y_pred, (size, size))
    Zp_err = np.reshape(MSE, (size, size))
    return Zp, Zp_err, x1x2


def gprocess(beam, plots=True):
    beam = str(beam).zfill(2)
    arr = np.loadtxt(f'/kutkin/beams/{beam}/result.dat')
    logging.info(arr.shape)
    x = arr[:,2]
    y = arr[:,3]
    z = arr[:,4]
    gp = gpr(x,y,z)
    gpfile = f'/kutkin/beams/{beam}/gpparams_all.dat'
    kparams = gp.kernel_.get_params()
    with open(gpfile, 'w') as inp:
        inp.write(f'kernel {gp.kernel_}\n')
        for key, val in kparams.items():
            inp.write(f'{key} {val}\n')
# predict:
    data, err, x1x2 = gppredict(gp)
    data += np.mean(z) # correct for mean
# scale
    logging.info('Original data range: {} -- {}'.format(data.min(), data.max()))
    logging.info('Scaling to [0,1]...')
    data = (data - np.nanmin(data))/(np.nanmax(data) - np.nanmin(data))

# interpolate -- works
    grid_x, grid_y = np.mgrid[-1536:1537, -1536:1537]
    data = interpolate.griddata(x1x2, data.ravel(), (grid_x, grid_y), method='linear')
    data = np.float32(data)
# create FITS file:
    header = fits.getheader(glob.glob(f'/kutkin/beams/{beam}/*.hdr')[0])
    hdu = fits.PrimaryHDU(data=data, header=header)
    hdu.writeto(f'/kutkin/beams/{beam}/pb{beam}.fits', overwrite=True)

    if plots:
        plt.subplot(221)
        plt.scatter(x, y, s=[10*i/max(z) for i in z])
        plt.plot(points[:,0], points[:,1], 'k.', ms=1)
        plt.title(f'Beam {beam}')
        plt.subplot(222)
        interval = ZScaleInterval()
        norm = ImageNormalize(data, interval=interval)
        X0p, X1p = x1x2[:,0].reshape(shape, shape), x1x2[:,1].reshape(shape,shape)
        plt.pcolormesh(X0p, X1p, Zp, vmin=-0.01, vmax=0.1, cmap='viridis')
        plt.title('GPR')
        plt.subplot(223)
        plt.imshow(data, origin='lower', vmin=-0.01, vmax=0.1, cmap='viridis')
        plt.contour(data, [0.1, 0.5], colors='white')
        plt.title('Linear interp')
        plt.subplot(224)
        plt.semilogy(np.hypot(x,y), z, 'o', ms=1)
        plt.axhline(0.1, ls='--', c='k', lw=0.5)
        plt.title('Radplot (all PA)')
        plt.gcf().set_size_inches(8, 8)
        plt.gcf().tight_layout()
        plt.gcf().savefig(f'/kutkin/beams/{beam}/gpr_all.png', dpi=150)
        plt.show()
    np.save(f'/kutkin/beams/{beam}/all_gp_interp.npy', data)
    return gp, data


def tid2dt(tid):
    return datetime.datetime.strptime(str(tid)[:6], '%y%m%d')


def make_pbeam(tid, beam, nnear=5, maxdt=None, clip=0.05, out=None):
    """
    Create primary beam model (FITS file)

    Parameters
    ----------
    tid : int or str
    beam : int or str
    nnear : int, optional
        number of near epochs to append. The default is 5.
    maxdt : int, optional
        maximal time separation of these ephochs (days). If set -- the nnear will be ignored.
    clip : float, optional
        crop the data behind this level. not implemented. The default is 0.05.
    Returns
    -------
    None.
    """

    beam = str(beam).zfill(2)
    tid = int(tid)
    date = tid2dt(tid)
    logging.info('Making pbeam for %s %s (date: %s)', tid, beam, date)
    df = pd.read_csv(f'/kutkin/beams/{beam}/result.dat', sep='\s+',
                      names=['tid', 'beam', 'x', 'y', 'z'])
    tids = sorted(set(df.tid.values))
    dates = np.array([tid2dt(_) for _ in tids])
    inds = np.argsort(abs(dates - date))
    tids_to_use = list(np.array(tids)[inds][:nnear])
    if maxdt is not None:
        cond = abs(dates - date) <= datetime.timedelta(maxdt)
        tids_to_use = list(np.array(tids)[cond])
        logging.info('Found %d observations within %d days. Processing...', len(tids_to_use), maxdt)
    else:
        logging.info('Found %d observations. Processing...', len(tids_to_use))
    # logging.debug(tids_to_use)
    group = df.query('tid in @tids_to_use')
    if group.empty:
        logging.error('No data!')
        return
    logging.info('%d points', len(group))
    x, y, z = group.x, group.y, group.z
    gp = gpr(x, y, z)
    shape = 750
    size = 100
    # shape = 500 # for driftscans
    # size = 40
    data, err, x1x2 = gppredict(gp, shape=shape, size=size)
    data += np.mean(z) # correct for mean
# scale
    logging.info('Original data range: {} -- {}'.format(data.min(), data.max()))
    logging.info('Scaling to [0,1]...')
    data = (data - np.nanmin(data))/(np.nanmax(data) - np.nanmin(data))

    data = np.float32(data)
    header = fits.getheader(glob.glob(f'/kutkin/beams/{beam}/*.hdr')[0])
    factor = 2.0 * shape / size # increaze of CDELT
    crdelt1 = header['CDELT1']
    header.update(NAXIS1=size, NAXIS2=size, CRPIX1=size/2, CRPIX2=size/2,
                  CDELT1=header['CDELT1']*factor,CDELT2=header['CDELT2']*factor)
    for k in ['HISTORY', 'BMAJ', 'BMIN', 'BPA', 'NITERS', 'OBJECT']:
        header.remove(k, remove_all=True)
    s = str(gp.kernel_).split('+')[0]
    header.update(ORIGIN=f'GPR: {s}')
# create FITS file:
    hdu = fits.PrimaryHDU(data=data, header=header)
    if out is None:
        out = os.path.join(f'{tid}_{beam}_pb.fits')
    hdu.writeto(out, overwrite=True)
    return gp, data, err, out


#% time evolution -- all within the errors. aug is a bit outlier...
def time_evolution():
    tids = [190815000,190915000,191015000,191115000,191215000,200115000]
    beam = 1
    res = []
    err = []
    for tid in tids:
        gp, data, err_i, out = make_pbeam(tid, beam, maxdt=14)
        res.append(data)
        err.append(err_i)
        gpsave(gp, f'/kutkin/test/time_evolution/gp_{tid}_{beam}.pkl')
    res = np.array(res)
    err = np.array(err)
    np.save('/kutkin/test/time_evolution/data.npy', res)
    np.save('/kutkin/test/time_evolution/err.npy', err)
### can be run separately:
    d = np.load('data.npy')
    e = np.load('err.npy')
    x = np.arange(len(d[0,...]))
    s = len(d[0,...]) // 2
    plt.subplot(121)
    for i, tid in enumerate(tids):
        plt.plot(d[i,:,s], label=int(tid/1000))
        if i == 3:
            plt.fill_between(x, d[i,:,s]-e[i,:,s], d[i,:,s]+e[i,:,s],alpha=0.3,color='gray')
    plt.legend()
    plt.title(f'Beam: {beam} (y-slice)')
    plt.subplot(122)
    for i, tid in enumerate(tids):
        plt.plot(d[i,s,:], label=int(tid/1000))
        if i == 3:
            plt.fill_between(x, d[i,s,:]-e[i,s,:], d[i,s,:]+e[i,s,:],alpha=0.3,color='gray')
    plt.legend()
    plt.title(f'Beam: {beam} (x-slice)')
    plt.show()


def pball(beam):

    gp, data, err, out = make_pbeam(191010000, beam, maxdt=365,
                                    out=f'/kutkin/beams/{beam}/pb_{beam}_gpall.fits')
    gpsave(gp, f'/kutkin/beams/{beam}/gpall.pkl')
    np.save(f'/kutkin/beams/{beam}/data_gpall.npy', data)
    np.save(f'/kutkin/beams/{beam}/err_gpall.npy', err)


# def pbplot(ffile):

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

#%% bulk process all the data to get the A/N relation
    # beams = [str(_).zfill(2) for _ in range(40)]
    # # for b in beams[::2]:
    # # for b in beams[::2][::-1]:
    # # for b in beams[1::2]:
    # for b in beams[1::2][::-1]:
    #     process(b)
#%% compare to drift scans -- the latter seem to be WRONG, e.g. beam 01 (see files in /kutkin/test/compare_gpbeams_with_driftscans)
    # tid = 191120000 # fake TID corresponding to drift scans date
    # beam = 1 # beam 1 differs most dramatically -- 90 degrees rotation?
    # out, data, err, gp = make_pbeam(tid, beam, maxdt=14)

#%% GPall (make pbeams for all beam images):
    # from multiprocessing import Pool
    # beams = [str(_).zfill(2) for _ in range(40)]
    # p = Pool(10)
    # p.map(pball, beams)
#%% mosaic plot
    beams = [str(_).zfill(2) for _ in range(40)]
    tid = 190915041
    path = '/kutkin/test/mosaic_with_gpbeams'
    for beam in beams:
        os.chdir(path)
        if os.path.exists(f'{path}/gp{beam}.pkl'):
            continue
        try:
            gp, data, err, out = make_pbeam(tid, beam, maxdt=14, out=f'{path}/{tid}_{beam}_pb.fits')
            gpsave(gp, f'{path}/gp{beam}.pkl')
            np.save(f'{path}/data{beam}.npy', data)
            np.save(f'{path}/err{beam}.npy', err)
        except Exception as e:
            logging.exception(e)
#%% test - passed

# img = '/home/kutkin/mnt/hyperion/kutkin/beams/00/test/191227013_00.fits'
# pb = '/home/kutkin/mnt/hyperion/kutkin/beams/00/test/pb00.fits'
# corr = fits_operation(img, pb, operation='/', out=img.replace('.fits','_corr.fits'))
# img, cat, rep = validation.run(corr, redo=True)

# validation.run('/home/kutkin/mnt/hyperion/kutkin/psfvar/190915041/190915041_12_mos.fits', redo=True)
# validation.run('/home/kutkin/mnt/hyperion/kutkin/psfvar/190915041/190915041_12.fits', redo=True)
