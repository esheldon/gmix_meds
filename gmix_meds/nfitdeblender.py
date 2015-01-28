from __future__ import print_function
import numpy
from .nfit import *
from ngmix import print_pars

from ngmix.fitting import MaxSimple, EIG_NOTFINITE

class MLDeblender(MedsFit):
    """
    Maximum Likelihood Deblender
    """
    
    def do_fits(self):
        """
        Fit all objects in our list
        """
        
        t0=time.time()
        num = len(self.fofids)
        
        for fofid in self.fofids:
            if self['save_obs_per_fof']:
                self.fof_mb_obs_list = {}
                
            #################################
            #first get fiducial models
            #################################
            
            fofmems = self.fofid2mindex[fofid]
            q = numpy.argsort(self.meds_list[0]['box_size'][fofmems])
            q = q[::-1]
            sfofmems = fofmems[q]
            
            for mindex in fofmems:
                if self.data['processed'][mindex] > 0:
                    continue
                
                print('index: %d:%d' % (mindex,len(fofmems)-1), )
                ti = time.time()
                self.fit_obj(mindex,model_neighbors=False)
                ti = time.time()-ti
                print('    time:',ti)
                
            
            #################################
            #now deblend the light
            #################################
                
            #if only 1 mem, no deblending needed!
            if len(fofmems) == 1:
                continue
                        
            for itr in range(self['max_deblend_itr']):
                #copy data
                self.prev_data = self.data.copy()
                
                rfofmems = numpy.random.permutation(fofmems)
                for mindex in rfofmems:
                    self.data['processed'][mindex] += 1
                    
                    print('index: %d:%d' % (mindex,len(rfofmems)-1), )
                    ti = time.time()                    
                    #skip if flags are set from first try
                    if self.data['flags'][mindex] > 0 and self.data['flags'][mindex] != NO_ATTEMPT:
                        pass
                    else:
                        self.fit_obj(mindex,model_neighbors=True)
                    ti = time.time()-ti
                    print('    time:',ti)
                    
                #FIXME do conv check
                
                tm=time.time()-t0
                self._try_checkpoint(tm) # only at certain intervals
                    
            #FIXME: do cov comp here...

            if self['save_obs_per_fof']:
                del self.fof_mb_obs_list
        
        tm=time.time()-t0
        print("time:",tm)
        print("time per fof:",tm/num)
        
    def _fit_all_models(self):
        """
        Fit psf flux and other models
        """
        dindex=self.dindex

        if self.data['flags'][dindex] == NO_ATTEMPT:        
            flags=0
            # fit both coadd and se psf flux if exists
            self._fit_psf_flux()

            s2n=self.data['psf_flux'][dindex,:]/self.data['psf_flux_err'][dindex,:]
            max_s2n=numpy.nanmax(s2n)

            if max_s2n >= self['min_psf_s2n'] and len(self['fit_models']) > 0:
                fit_obj = True
            else:
                fit_obj = False
                mess="    psf s/n too low: %s (%s)" % (max_s2n,self['min_psf_s2n'])
                print(mess)            
                flags |= LOW_PSF_FLUX
        else:
            #we don't get here if flags != 0 from previous attempt
            #test is in do_fits above
            fit_obj = True
            flags = self.data['flags'][dindex]
        
        if fit_obj:
            for model in self['fit_models']:
                print('    fitting:',model)

                if self['fit_coadd_galaxy']:
                    print('    coadd')
                    self._run_model_fit(model, coadd=True)

                if self['fit_me_galaxy']:
                    print('    multi-epoch')
                    self._run_model_fit(model, coadd=False)

        return flags

    def _run_model_fit(self, model, coadd=False):
        """
        wrapper to run fit, copy pars, maybe make plots
        sets .fitter or .coadd_fitter
        """
        dindex = self.mindex
        
        if coadd:
            if self.data['processed'][dindex] == 0:
                self.guesser = self._get_guesser('coadd_psf')
            else:
                pars = self.data['coadd_'+model+'_pars'][dindex]
                self.guesser = FixedParsGuesser(pars,pars*self['deblend_guess_frac'])
            mb_obs_list=self.sdata['coadd_mb_obs_list']
        else:
            if self.data['processed'][dindex] == 0:
                self.guesser = self._get_guesser('me_psf')
            else:
                pars = self.data[model+'pars'][dindex]
                self.guesser = FixedParsGuesser(pars,pars*self['deblend_guess_frac'])
            mb_obs_list=self.sdata['mb_obs_list']

        fitter=self._fit_model(mb_obs_list, model)

        self._copy_simple_pars(fitter, coadd=coadd)

        self._print_res(fitter, coadd=coadd)

        if self['make_plots']:
            self._do_make_plots(fitter, model, coadd=coadd)

        if coadd:
            self.coadd_fitter=fitter
        else:
            self.fitter=fitter

    def _fit_model(self, mb_obs_list, model):
        """
        Fit all the simple models
        """

        fitter=self. _fit_simple_max(self, mb_obs_list, model, self['nm_pars'], ntry=1)
        
        #FIXME - not doing MCMC stats clearly - any side effects?
        # also adds .weights attribute
        #self._calc_mcmc_stats(fitter, model)

        return fitter
        
    def _fit_simple_max(self, mb_obs_list, model, nm_pars, ntry=1):
        """
        parameters
        ----------
        mb_obs_list: MultiBandObsList
            The observations to fit
        model: string
            model to fit
        params: dict
            from the config file
        """

        prior=self['model_pars'][model]['gflat_prior']

        method = 'Nelder-Mead'

        fitter=MaxSimple(mb_obs_list,
                         model,
                         prior=prior,
                         method=method)

        guess=self.guesser(prior=prior)

        for i in xrange(ntry):
            fitter.run_max(guess, **nm_pars)
            res=fitter.get_result()

            if (ntry > 1) and (res['flags'] & EIG_NOTFINITE) != 0:
                # bad covariance matrix, need to get a new guess
                print_pars(res['pars'],front='        bad cov at pars')
                self.guesser=FromParsGuesser(guess, guess*0.1)
                guess=self.guesser(prior=prior)
                print_pars(guess,front='        new guess')
            else:
                break

        return fitter
