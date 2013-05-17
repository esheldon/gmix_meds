from sys import stderr
import os
import numpy
from numpy import sqrt
import gmix_image
from gmix_image import GMixEMBoot
import psfex
import fitsio

import images

PIXEL_SCALE=0.265

def get_outdir():
    desdata=os.environ['DESDATA']
    return os.path.join(desdata, 'users','esheldon','double-psf')

def get_image_dir():
    d=get_outdir()
    return os.path.join(d, 'images')

def get_png_name(expname, ccd):
    d=get_image_dir()
    name='%s_%02d_psf.png' % (expname,ccd)
    return os.path.join(d, name)

def extract_ccd_flist(flist,ccd):
    """
    extract files with the requested ccd
    """
    pattern='_%02d_psfcat.psf' % ccd
    flist_keep=[]
    for f in flist:
        if pattern in f:
            flist_keep.append(f)
    return flist_keep


def fit_em2(flist, out_file, pos=[500.63, 600.25]):
    """
    Process the list, measureing a double gaussian
    and recording the center offsets
    """

    nf=len(flist)

    ngauss=2
    maxtry=10

    reslist=[]

    explist,ccdlist=extract_expname_ccd(flist)

    max_flen=get_max_slen(flist)
    max_explen=get_max_slen(explist)

    pnglist=[]
    for expname,ccd in zip(explist,ccdlist):
        png_path=get_png_name(expname,ccd)
        pnglist.append(png_path)
    max_pnglen=get_max_slen(pnglist)

    dt=[('fname','S%d' % max_flen),
        ('pngname','S%d' % max_pnglen),
        ('expname','S%d' % max_explen),
        ('ccd','i2'),
        ('flags','i2'),
        ('fwhm_arcsec','f8'),
        ('offset_arcsec','f8')]
    print dt

    data=numpy.zeros(nf, dtype=dt)

    for i in xrange(nf):

        psf_path=flist[i]
        expname=explist[i]
        png_path=pnglist[i]
        ccd=ccdlist[i]

        print >>stderr,'%d/%d' % (i+1,nf)
        print >>stderr,psf_path

        data['fname'][i] = psf_path
        data['expname'][i] = expname
        data['ccd'][i] = ccd

        pex=psfex.PSFEx(psf_path)
        fwhm_arcsec=pex.get_fwhm()*PIXEL_SCALE

        im,gm,flags,offset_arcsec=_do_measure_em2(pex,pos,ngauss,maxtry)

        data['flags'][i] = flags
        data['fwhm_arcsec'][i] = fwhm_arcsec
        data['offset_arcsec'][i] = offset_arcsec

        if flags != 0:
            continue

        data['pngname'][i]=png_path

        _do_plots(im, gm, expname, ccd, pos, offset_arcsec, png_path)

    print >>stderr,'writing output:',out_file
    fitsio.write(out_file, data, clobber=True)


def _do_measure_em2(pex, pos, ngauss, maxtry):
    im=pex.get_rec(pos[0], pos[1])

    cen_guess=pex.get_center(pos[0], pos[1])
    sigma_guess=pex.get_sigma()

    gm=GMixEMBoot(im, ngauss, cen_guess,
                  sigma_guess=sigma_guess,
                  maxtry=maxtry)

    res=gm.get_result()
    flags=res['flags']

    offset_arcsec=-9999.
    if flags != 0:
        print >>stderr,'    failed to fit:',flags
    else:

        gmix=gm.get_gmix()
        dlist=gmix.get_dlist()

        offset=sqrt( (dlist[0]['row']-dlist[1]['row'])**2 + 
                     (dlist[0]['col']-dlist[1]['col'])**2 )
        offset_arcsec=offset*PIXEL_SCALE

    return im,gm,flags,offset_arcsec

def calc_offset_arcsec(gmix):
    dlist=gmix.get_dlist()

    offset=sqrt( (dlist[0]['row']-dlist[1]['row'])**2 + 
                 (dlist[0]['col']-dlist[1]['col'])**2 )
    offset_arcsec=offset*PIXEL_SCALE
    return offset_arcsec


def _do_plots(im, gm, expname, ccd, pos, offset, png_path):
    model=gm.get_model()
    model *= im.sum()/model.sum()
    plt=images.compare_images(im*100, model*100,
                              label1='image',
                              label2='model',
                              nonlinear=.2,
                              show=False)

    title="%s_%02d row: %.1f col: %.2f offset: %.2f''"
    title=title % (expname,ccd,pos[0],pos[1],offset)
    plt.title=title
    plt.title_style['fontsize']=2

    print >>stderr,'    ',png_path
    plt.write_img(1000,1000,png_path)

def extract_expname_ccd(flist):
    explist=[]
    ccdlist=[]

    for f in flist:
        bname=os.path.basename(f)
        expname=bname[0:-14]
        ccd=int( bname[-13:-11] )

        explist.append(expname)
        ccdlist.append(ccd)

    return explist, ccdlist

def group_by_exposure(flist):
    expdict={}
    explist,ccdlist=extract_expname_ccd(flist)

    nf=len(flist)
    for i in xrange(nf):
        fname=flist[i]
        expname=explist[i]
        if expname in expdict:
            expdict[expname].append(fname)
        else:
            expdict[expname] = [fname]

    return expdict

def get_max_slen(slist):
    max_slen=0
    for f in slist:
        slen=len(f)
        if slen > max_slen:
            max_slen=slen

    return max_slen

