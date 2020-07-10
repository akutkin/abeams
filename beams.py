#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Try to restore primary beam shape by comparison source fluxes with NVSS

Created on Sat Jun 13 16:26:13 2020

@author: kutkin
"""
# import matplotlib
# matplotlib.use('Agg')


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

import shutil
from atools.utils import fits_transfer_coordinates, fits_squeeze, fits_crop, fits_clip
from reproject.mosaicking import reproject_and_coadd, find_optimal_celestial_wcs
from reproject import reproject_interp
from multiprocessing import Pool

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
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

from atools.utils import fits_reconvolve_psf, get_tid_beam, fits_operation, fits_transpose
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
        result = f'/kutkin/beams/{beam}/APERTIF_NVSS_relation.dat'

        # if os.path.exists(result):
        #     logging.debug('Result file exists %s', result)
        #     continue
        if os.path.exists(commoncat) and os.path.exists(hdrfile):
            df = pd.read_csv(commoncat)
            wcs = WCS(hdrfile).dropaxis(-1).dropaxis(-1)
        else:
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
            hdr.tofile(hdrfile)
            os.remove(tmp)
        radec = [list(df.RA_1.values), list(df.DEC_1.values)]
        xi, yi = wcs.all_world2pix(radec[0], radec[1], 1)
        xi = [_-1537 for _ in xi]
        yi = [_-1537 for _ in yi]
        zi = list(df.Total_flux_1.values/df.Total_flux_2.values)
        x += xi
        y += yi
        z += zi
        with open(result, 'a+') as out:
            for i,j,k in zip(xi,yi,zi):
                out.write(f'{tid} {beam} {i} {j} {k}\n')
        # print(df)
    return


def gpr(x,y,z):
    """
    note the narrow priors -- l=340 ~ 22 arcmin for apercal (!) images
    """
    zmean = np.mean(z)
    kernel = ConstantKernel(0.2**2) * RBF([330, 330], (200, 500)) + \
              ConstantKernel(0.05**2) * RBF([0.1, 0.1], (1e-2, 100)) + WhiteKernel(0.001)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=25)
    logging.info('Gaussian processing')
    logging.info('Starting kernel %s', gp.kernel)
    X = np.stack((y,x)).T
    gp.fit(X, z-zmean)
    logging.info('Learned kernel %s', gp.kernel_)
    return gp


def gpsave(gp, fname):
    with open(fname, 'wb') as inp:
        pkl.dump(gp, inp)


def gpload(fname):
    with open(fname, 'rb') as out:
        return pkl.load(out)


def gppredict(gp, shape=750, size=100, normalize=True):
    x1 = np.linspace(-shape, shape, size) #p
    x2 = np.linspace(-shape, shape, size) #q
    x1x2 = np.array(list(product(x1, x2)))
# Use the prediction from the major part of the kernel:
    if 'k2__noise_level' in gp.kernel_.get_params(): # if white kernell is used...
        gp.kernel_.set_params(k2=ConstantKernel(0), k1__k2=ConstantKernel(0))
    else:
        gp.kernel_.set_params(k2=ConstantKernel(0.0))
    y_pred, MSE = gp.predict(x1x2, return_std=True)
    arr = np.reshape(y_pred, (size, size))
    err = np.reshape(MSE, (size, size))
    if normalize:
        newarr = (arr - np.nanmin(arr)) / (np.nanmax(arr) - np.nanmin(arr))
# TODO: check this
        newerr = err / (np.nanmax(arr) - np.nanmin(arr))
        return newarr, newerr, x1x2
    else:
        return arr, err, x1x2


def tid2dt(tid):
    return datetime.datetime.strptime(str(tid)[:6], '%y%m%d')


def _data_within_dt(tid, beam, dt):

    date = tid2dt(tid)
    df = pd.read_csv(f'/kutkin/beams/{beam}/APERTIF_NVSS_relation.dat', sep='\s+',
                      names=['tid', 'beam', 'x', 'y', 'z'])
    tids = sorted(set(df.tid.values))
    dates = np.array([tid2dt(_) for _ in tids])
    cond = abs(dates - date) <= datetime.timedelta(dt)
    tids_to_use = list(np.array(tids)[cond])
    group = df.query('tid in @tids_to_use')
    return group


def make_pbeam(tid, beam, maxdt=7, nnear=None, clip=0.05, shape=750, size=100,
               gpfile=None, save_gp_to=None, out=None):
    """
    Create primary beam model (FITS file)

    Parameters
    ----------
    tid : int or str
    beam : int or str
    nnear : int, optional
        number of near epochs to append. Ignored if maxdt is set. The default is 5.
    maxdt : int, optional
        maximal time separation of these ephochs (days). If set -- the nnear will be ignored.
    clip : float, optional
        crop the data behind this level. not implemented. The default is 0.05.
    shape : int, optional
        the shape to predict GP on (full image is 1536, driftscan pbeams are 500... 750 should include 1st null level
    size : int, optionsl
        the size of the output array (NAXIS), driftscans imgaes have 40...
    Returns
    -------
    None.
    """

    beam = str(beam).zfill(2)
    tid = int(tid)
    logging.info('Making pbeam for %s %s (date: %s)', tid, beam, tid2dt(tid))

# GP
    if gpfile is not None:
        logging.info('Loading GP-file: %s', gpfile)
        gp = gpload(gpfile)
    else:
        group = _data_within_dt(tid, beam, maxdt)
        i = 0
        while len(group) < 500:
            i += 1
            logging.warning("Too few data points for GPR... Increasing time interval by %d days", i)
            group = _data_within_dt(tid, beam, maxdt+i)
        logging.info('Processing %d sources within %d days', len(group), maxdt+i)

        x, y, z = group.x, group.y, group.z
        gp = gpr(x, y, z)

# save the full GP:
    if save_gp_to is not None:
        gpsave(gp, f'{save_gp_to}')
# the following modifies GP:
    data, err, x1x2 = gppredict(gp, shape=shape, size=size, normalize=True)
    data = np.float32(data)

    header = fits.getheader(glob.glob(f'/kutkin/beams/{beam}/*.hdr')[0])
    factor = 2.0 * shape / size # increaze of CDELT

    header.update(NAXIS1=size, NAXIS2=size, CRPIX1=size/2, CRPIX2=size/2,
                  CDELT1=header['CDELT1']*factor, CDELT2=header['CDELT2']*factor)
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


def _go(beam):
    # these for the coordinates:
    tid = 190915041
    fitsfiles = glob.glob(f'/kutkin/images_apercal/{tid}_{beam}*.fits') or \
               glob.glob(f'/kutkin/acat/dr1/{tid}_{beam}.fits')
    fitsfile = fitsfiles[0]
    # pb = f'/kutkin/beams/{beam}/pb_{beam}_gpall.fits'
# make new fits:
    gpfile = f'/kutkin/beams/{beam}/gpall.pkl'
    good_res_file = f'/kutkin/beams/mosaic/{beam}.fits'
    if not os.path.exists(good_res_file):
        gp, data, err, tmpfile = make_pbeam(tid=tid, beam=beam, shape=1000,
                                            size=500, gpfile=gpfile,
                                            out=good_res_file)
    tmpfile = f'/kutkin/beams/mosaic/{beam}_tmp.fits'
    shutil.copyfile(good_res_file, tmpfile)

# squeeze, transfer coordinates crop, clip:
    tmpfile = fits_squeeze(tmpfile)
    fits_transfer_coordinates(fitsfile, tmpfile)
    tmpfile, _ = fits_crop(tmpfile, level=0.05)
    tmpfile = fits_clip(tmpfile, level=0.15, out=tmpfile)
    return tmpfile


def mosaic_pbeams():

    beams = [str(_).zfill(2) for _ in range(40)]
    # p = Pool(1)
    # p.map(_go, beams)
    # pbs = glob.glob('/kutkin/beams/mosaic/*_tmp.fits')
    pbs = []
    for beam in beams:
        pbs.append(_go(beam))

    clip=0.15 # change above
    wcs_out, shape_out = find_optimal_celestial_wcs(pbs, auto_rotate=False)
    array, footprint = reproject_and_coadd(pbs, wcs_out, shape_out=shape_out,
                                            reproject_function=reproject_interp,
                                            combine_function='sum')
    array = np.float32(array)
    hdr = wcs_out.to_header()
    fits.writeto('/kutkin/beams/mosaic/pbmosaic{:d}.fits'.format(int(clip*100)), data=array,
                  header=hdr, overwrite=True)


def pball(beam):

    gp, data, err, out = make_pbeam(191010000, beam, maxdt=365,
                                    out=f'/kutkin/beams/{beam}/pb_{beam}_gpall.fits',
                                    save_gp_to=f'/kutkin/beams/{beam}/gpall.pkl')
    # np.save(f'/kutkin/beams/{beam}/data_gpall.npy', data)
    # np.save(f'/kutkin/beams/{beam}/err_gpall.npy', err)


# def pbplot(ffile):

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    # gp1, data1, err1, out = make_pbeam(200101000, 7, maxdt=7, clip=0.05, out=None)
    # gp2, arr2, err2, out = make_pbeam(190915041, 1, maxdt=7, clip=0.05, out=None)


#%% bulk process all the data to get the A/N relation
    # from multiprocessing import Pool
    # beams = [str(_).zfill(2) for _ in range(40)]
    # p = Pool(40)
    # p.map(process, beams)
# use "for" if bdsf involved...
    # for b in beams[::2]:
    # for b in beams[::2][::-1]:
    # for b in beams[1::2]:
    # for b in beams[1::2][::-1]:
        # process(b)
#%% compare to drift scans -- the latter seem to be WRONG, e.g. beam 01 (see files in /kutkin/test/compare_gpbeams_with_driftscans)
    # tid = 191120000 # fake TID corresponding to drift scans date
    # beam = 1 # beam 1 differs most dramatically -- 90 degrees rotation?
    # out, data, err, gp = make_pbeam(tid, beam, maxdt=14)

#%% GPall (make pbeams for all beam images):
    # from multiprocessing import Pool
    # beams = [str(_).zfill(2) for _ in range(40)]
    # p = Pool(len(beams))
    # p.map(pball, beams)

#%% mosaic plot -- works...
    # mosaic_pbeams()

#%% plots
#    see figures script in acat
        # print(gp)




#%% test - passed

# img = '/home/kutkin/mnt/hyperion/kutkin/beams/00/test/191227013_00.fits'
# pb = '/home/kutkin/mnt/hyperion/kutkin/beams/00/test/pb00.fits'
# corr = fits_operation(img, pb, operation='/', out=img.replace('.fits','_corr.fits'))
# img, cat, rep = validation.run(corr, redo=True)

# validation.run('/home/kutkin/mnt/hyperion/kutkin/psfvar/190915041/190915041_12_mos.fits', redo=True)
# validation.run('/home/kutkin/mnt/hyperion/kutkin/psfvar/190915041/190915041_12.fits', redo=True)
