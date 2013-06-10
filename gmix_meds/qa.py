import os
import numpy
from numpy import log10
import fitsio

SCALE=0.265

def do_many_compare_psfmag(goodlist_file):
    import json
    import desdb

    with open(goodlist_file) as fobj:
        data=json.load(fobj)

    tdict = key_by_tile_band(data)

    run=data[0]['run']
    
    desdata=os.environ['DESDATA']
    outdir=os.path.join(desdata,'users','esheldon','gfme-qa',run)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    print outdir

    df=desdb.DESFiles()
    for tileband,fdict in tdict.iteritems():

        fname=df.url(type='wlpipe_me_generic',
                     run=run,
                     tilename=fdict['tilename'],
                     band=fdict['band'],
                     filetype='lmfit',
                     ext='fits')

        print fname

        comp=ComparePSFMags(fname)
        plt=comp.doplot()
        plt_stars=comp.doplot(stars=True)

        bname=os.path.basename(fname)
        epsname=bname.replace('.fits','-magdiff.eps')
        epsname=os.path.join(outdir,epsname)

        epsname_stars=epsname.replace(".eps", "-stars.eps")

        print '    ',epsname
        print '    ',epsname_stars

        plt.write_eps(epsname)
        plt_stars.write_eps(epsname_stars)

def key_by_tile_band(data):
    odict={}

    for d in data:
        tilename=d['tilename']
        band=d['band']

        key='%s-%s' % (tilename,band)

        odict[key] = d

    return odict


class FluxHist(object):
    """
    Plot the histogram of flux or magnitude
    """
    def __init__(self, fname, scale=SCALE):
        self.fname=fname
        self.scale=scale

        self._load_data()

    def doplot(self, model, show=False, domag=False, **keys):
        import esutil as eu
        
        flagn='%s_flags' % model
        fluxn='%s_flux' % model

        data=self.data
        if (flagn not in data.dtype.names
            or fluxn not in data.dtype.names):
            raise ValueError("model not found: '%s'" % model)

        w,=numpy.where( (data[flagn]==0) & (data[fluxn] > 0.001) )

        if domag:
            fscale = data[fluxn][w]/( self.scale**2 )
            hdata = -2.5*log10(fscale) + self.meta['magzp_ref'][0]
            xlabel=r'$mag_{%s}' % model
            xlog=False
        else:
            hdata = log10( data[fluxn][w] )
            xlabel=r'$log_{10}(F_{%s})$' % model
            xlog=True
        
        std=hdata.std()
        mn=hdata.mean()

        nsig=5
        xmin = max( mn-nsig*std, hdata.min() )
        xmax = min( mn+nsig*std, hdata.max() )
        xrng=[xmin,xmax]

        plt=eu.plotting.bhist(hdata, binsize=0.1*std, show=show,
                              xrange=xrng,
                              xlabel=xlabel, xlog=xlog)

        return plt
        
    def _load_data(self):
        with fitsio.FITS(self.fname) as fobj:
            self.data=fobj['model_fits'][:]
            self.meta=fobj['meta_data'][:]

class CompareBase(object):
    """
    Compare to sextractor psf mags
    """
    def __init__(self, fname, coadd_cat_file=None, scale=SCALE):
        self.fname=fname
        self.coadd_cat_file=coadd_cat_file
        self.scale=scale

        self._load_data()

        
    def _load_data(self):
        with fitsio.FITS(self.fname) as fobj:
            self.data=fobj['model_fits'][:]
            self.meta=fobj['meta_data'][:]

        if self.coadd_cat_file is not None:
            coadd_cat_file=self.coadd_cat_file
        else:
            coadd_cat_file=self.meta['coaddcat_file'][0]
        self.coaddcat_data=fitsio.read(coadd_cat_file,lower=True)

class ComparePSFMags(CompareBase):
    """
    Compare to sextractor psf mags
    """
    def __init__(self, fname, coadd_cat_file=None, scale=SCALE):
        super(ComparePSFMags,self).__init__(fname,
                                            coadd_cat_file=coadd_cat_file,
                                            scale=scale)

    def doplot(self, show=False, stars=False, 
               star_spread_model_max=0.002, **keys):
        import esutil as eu
        import biggles
        

        data=self.data
        coaddcat_data=self.coaddcat_data

        logic= ( (data['psf_flags']==0) 
                & (data['psf_flux'] > 0.001)
                & (data['psf_flux_err'] > 0.001)
                & (coaddcat_data['flags']==0) )
        if stars:
            av=numpy.abs(coaddcat_data['spread_model'])
            logic = logic & (av < star_spread_model_max)

        w,=numpy.where(logic)

        fs2n = data['psf_flux'][w]/data['psf_flux_err'][w]

        ww,=numpy.where(fs2n > 2)
        w=w[ww]

        fscale = data['psf_flux'][w]/( self.scale**2 )
        mag = -2.5*log10(fscale) + self.meta['magzp_ref'][0]
        sxmag = coaddcat_data['mag_psf'][w]

        mdiff = mag - sxmag

        xlabel=r'$mag^{SX}_{psf}$'
        vs_ylabel=r'$mag^{ME}_{psf}$'
        diff_ylabel=r'$mag^{ME}_{psf} - mag^{SX}_{psf}$'

        tab=biggles.Table(1,2)

        tab.title=os.path.basename(self.fname).replace('.fits','')
        tab.aspect_ratio=1./1.618
        type='dot'
        xrng=[15,26]
        plt_vs=eu.plotting.bscatter(mag, sxmag,
                                    type=type,
                                    xrange=xrng,
                                    yrange=xrng,
                                    xlabel=xlabel,
                                    ylabel=vs_ylabel,
                                    show=False)

        plt_diff=eu.plotting.bscatter(mag, mdiff,
                                      type=type,
                                      xrange=xrng,
                                      yrange=[-0.3,0.35],
                                      xlabel=xlabel,
                                      ylabel=diff_ylabel,
                                      show=False)

        plt_vs.aspect_ratio=1
        plt_diff.aspect_ratio=1

        if stars:
            mlow=16
            mhigh=20
            wfit,=numpy.where( (sxmag >  mlow) & (sxmag < mhigh) )
            coeffs=numpy.polyfit(sxmag[wfit], mdiff[wfit], 0)
            c=biggles.Curve(xrng,[coeffs[0],coeffs[0]])
            plt_diff.add(c)

            labstr=r'$spread_{model} < %.3f$' % star_spread_model_max
            lab=biggles.PlotLabel(0.05,0.9,labstr,halign='left')
            plt_vs.add(lab)

        tab[0,0] = plt_vs
        tab[0,1] = plt_diff
        if show:
            tab.show()
        return tab
        
class CompareSimpleMags(CompareBase):
    """
    Compare to sextractor psf mags
    """
    def __init__(self, fname, coadd_cat_file=None, scale=SCALE):
        super(CompareSimpleMags,self).__init__(fname,
                                               coadd_cat_file=coadd_cat_file,
                                               scale=scale)

    def doplot(self, model, show=False, 
               star_spread_model_max=0.002, **keys):
        import esutil as eu
        import biggles
        

        data=self.data
        coaddcat_data=self.coaddcat_data

        flags=data['%s_flags' % model]
        flux=data['%s_flux' % model]
        flux_err=data['%s_flux_err' % model]

        logic= ( (flags==0) 
                & (flux > 0.001)
                & (flux_err > 0.001)
                & (coaddcat_data['flags']==0)
                & (coaddcat_data['spread_model'] > star_spread_model_max) )

        w,=numpy.where(logic)

        fscale = flux[w]/( self.scale**2 )
        mag = -2.5*log10(fscale) + self.meta['magzp_ref'][0]
        mag_auto = coaddcat_data['mag_auto'][w]
        mag_auto_err = coaddcat_data['magerr_auto'][w]

        xlabel=r'$mag^{SX}_{auto} or mag^{LM}_{%s}$' % model
        ylabel=r'$\sigma( mag^{SX}_{auto} )$'

        xrng=[15,26]
        yrng=[0,0.5]
        type='dot'
        size=1.5
        plt0=eu.plotting.bscatter(mag_auto, mag_auto_err,
                                  xrange=xrng,
                                  yrange=yrng,
                                  xlabel=xlabel,
                                  ylabel=ylabel,
                                  type=type,
                                  size=size,
                                  show=False)

        plt=eu.plotting.bscatter(mag, mag_auto_err,
                                 type=type,
                                 size=size,
                                 color='red',
                                 plt=plt0,
                                 show=False)



        plt.title=os.path.basename(self.fname).replace('.fits','')


        plt.aspect_ratio=1


        if show:
            plt.show()
        return plt
        


class FluxVsFluxErr(CompareBase):
    """
    Plot Flux vs Flux_err
    """
    def __init__(self, fname, coadd_cat_file=None, scale=SCALE):
        super(FluxVsFluxErr,self).__init__(fname,
                                               coadd_cat_file=coadd_cat_file,
                                               scale=scale)

    def doplot(self, model, show=False, 
               star_spread_model_max=0.002, **keys):
        import esutil as eu
        import biggles
        

        data=self.data
        coaddcat_data=self.coaddcat_data

        flags=data['%s_flags' % model]
        flux=data['%s_flux' % model]
        flux_err=data['%s_flux_err' % model]

        logic= ( (flags==0) 
                & (flux > 0.001)
                & (flux_err > 0.001) )

        w,=numpy.where(logic)

        fscale = flux[w]/( self.scale**2 )
        fscale_err = flux_err[w]/( self.scale**2 )

        log_fscale = numpy.log10( fscale )

        xlabel=r'$log_{10}( Flux_{%s}$ )' % model
        ylabel=r'$\sigma( Flux_{%s} )$' % model

        xrng=[1,6]
        yrng=[0,1000]
        type='dot'
        size=1.5
        plt=eu.plotting.bscatter(log_fscale, fscale_err,
                                 xrange=xrng,
                                 yrange=yrng,
                                 xlabel=xlabel,
                                 ylabel=ylabel,
                                 type=type,
                                 size=size,
                                 show=False)


        plt.title=os.path.basename(self.fname).replace('.fits','')


        plt.aspect_ratio=1


        if show:
            plt.show()
        return plt
        
