from __future__ import print_function
import numpy
from .nfit import *

class MHMedsFitModelNbrs(MHMedsFitHybrid):
    """
    Models Nbrs
    """

    def _fit_all_models(self):
        """
        Fit psf flux and other models
        """
        
        #default to true just in case
        if 'fit_coadd_galaxy' not in self:
            self['fit_coadd_galaxy'] = True
        
        mindex_local = self.mindex #index in current meds file
        meds = self.meds_list[0]
        number = meds['number'][mindex_local] #number for seg map, index+1 into entire meds file
        mindex_global = number-1
        
        flags=0
        # fit both coadd and se psf flux if exists
        self._fit_psf_flux()

        dindex=self.dindex
        s2n=self.data['coadd_psf_flux'][dindex,:]/self.data['coadd_psf_flux_err'][dindex,:]
        max_s2n=numpy.nanmax(s2n)

        if max_s2n >= self['min_psf_s2n'] and len(self['fit_models']) > 0:
            for model in self['fit_models']:
                print('    fitting:',model)
                
                if self['fit_coadd_galaxy']:
                    print('    coadd')
                    self._run_model_fit(model, self['coadd_fitter_class'], mindex_global, coadd=True)

                if self['fit_me_galaxy']:
                    print('    multi-epoch')
                    # fitter class should be mh...
                    self._run_model_fit(model, self['fitter_class'], mindex_global, coadd=False)

        else:
            mess="    psf s/n too low: %s (%s)"
            mess=mess % (max_s2n,self['min_psf_s2n'])
            print(mess)
            
            flags |= LOW_PSF_FLUX

        return flags

    def _run_model_fit(self, model, fitter_type, mindex_global, coadd=False):
        """
        wrapper to run fit, copy pars, maybe make plots

        sets .fitter or .coadd_fitter

        this one does not currently use self['guess_type']
        """
        
        nmod = Namer(model)        
        if coadd:
            n = Namer('coadd')
            pars = self.model_data['model_fits'][n(nmod('pars_best'))][mindex_global]
            pars_cov = self.model_data['model_fits'][n(nmod('pars_cov'))][mindex_global]
            pars_err = numpy.array([numpy.sqrt(pars_cov[i,i]) for i in xrange(len(pars))])
            self.guesser=FromFullParsGuesser(pars,pars_err)
            mb_obs_list=self.sdata['coadd_mb_obs_list']
        else:
            pars = self.model_data['model_fits'][nmod('pars_best')][mindex_global]
            pars_cov = self.model_data['model_fits'][nmod('pars_cov')][mindex_global]
            pars_err = numpy.array([numpy.sqrt(pars_cov[i,i]) for i in xrange(len(pars))])
            self.guesser=FromFullParsGuesser(pars,pars_err)
            mb_obs_list=self.sdata['mb_obs_list']

        fitter=self._fit_model(mb_obs_list,
                               model,
                               fitter_type=fitter_type)

        self._copy_simple_pars(fitter, coadd=coadd)

        self._print_res(fitter, coadd=coadd)

        if self['make_plots']:
            self._do_make_plots(fitter, model, coadd=coadd, fitter_type=fitter_type)

        if coadd:
            self.coadd_fitter=fitter
        else:
            self.fitter=fitter

