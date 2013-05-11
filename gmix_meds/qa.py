import numpy
from numpy import log10
import fitsio

class FluxHist(object):
    """
    Plot the histogram of flux or magnitude
    """
    def __init__(self, fname, scale=0.27):
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

class ComparePSFMags(object):
    """
    Compare to sextractor psf mags
    """
    def __init__(self, fname, scale=0.27):
        self.fname=fname
        self.scale=scale

        self._load_data()

    def doplot(self, show=False, **keys):
        import esutil as eu
        import biggles
        

        data=self.data
        sxdata=self.sxdata

        w,=numpy.where(  (data['psf_flags']==0) 
                       & (data['psf_flux'] > 0.001)
                       & (data['psf_flux_err'] > 0.001)
                       & (sxdata['flags']==0) )

        fs2n = data['psf_flux'][w]/data['psf_flux_err'][w]

        ww,=numpy.where(fs2n > 2)
        w=w[ww]

        fscale = data['psf_flux'][w]/( self.scale**2 )
        mag = -2.5*log10(fscale) + self.meta['magzp_ref'][0]

        mdiff = mag - sxdata['mag_psf'][w]

        xlabel=r'$mag^{SX}_{psf}'
        ylabel=r'$mag^{ME}_{psf} - mag^{SX}_{psf}$'

        tab=biggles.Table(2,1)

        plt_vs=eu.plotting.bscatter(mag, sxdata['mag_psf'][w],
                                 type='dot',
                                 xrange=[14,26],
                                 yrange=[14,26],
                                 xlabel=xlabel,
                                 ylabel=ylabel,
                                 show=False)

        plt_diff=eu.plotting.bscatter(mag, mdiff,
                                 type='dot',
                                 xrange=[14,26],
                                 yrange=[-0.1,0.2],
                                 xlabel=xlabel,
                                 ylabel=ylabel,
                                 show=False)

        tab[0,0] = plt_vs
        tab[1,0] = plt_diff
        if show:
            tab.show()
        return tab
        
    def _load_data(self):
        with fitsio.FITS(self.fname) as fobj:
            self.data=fobj['model_fits'][:]
            self.meta=fobj['meta_data'][:]

        sxname=self.meta['coaddcat_file'][0]
        self.sxdata=fitsio.read(sxname,lower=True)
