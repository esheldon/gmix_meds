from __future__ import print_function
import numpy
from .nfit import *
from ngmix import print_pars

from ngmix.fitting import MaxSimple, EIG_NOTFINITE

class MLDeblender(MedsFit):
    """
    Maximum Likelihood Deblender
    """
    
    def __init__(self,self,
                 conf,
                 priors,
                 meds_files,
                 fof_range=None,
                 model_data=None,
                 checkpoint_file=None,
                 checkpoint_data=None):
        self.fof_range = fof_range
        super(MLDeblender,self).__init__(self,
                                         conf,
                                         priors,
                                         meds_files,
                                         obj_range=None,
                                         model_data=model_data,
                                         checkpoint_file=checkpoint_file,
                                         checkpoint_data=checkpoint_data)
        self._build_fofmems_hash()
        
        
    def _build_fofmems_hash(self):
        #FIXME: fill this in!
        pass
        
    def do_fits(self):
        """
        Fit all objects in our list
        """

        last=self.index_list[-1]
        num=len(self.index_list)

        #################################
        #first get fiducial models
        #################################
        
        t0=time.time()
        
        #FIXME sort by size or mag
        for dindex in xrange(num):
            if self.data['processed'][dindex] > 0:
                continue

            mindex = self.index_list[dindex]
            print('index: %d:%d' % (mindex,last), )
            ti = time.time()
            
            #saving this bit of code for later
            self.fit_obj(dindex)
            ti = time.time()-ti
            print('    time:',ti)
            
        tm=time.time()-t0
        print("time:",tm)
        print("time per:",tm/num)
        
        #################################
        #now deblend the light
        #################################
        
        #FIXME: loop through fofs, only run fofs with 1 object once, so skip them
        fofids = np.unique(self.model_data['fofs']['fofid'])
        for fofid in fofids:
            fofmems = self.fofmem_hash[fofid]
            for itr in range(self['max_deblend_itr']):
                #copy data
                self.prev_data = self.data.copy()
                
                rfofmems = numpy.random.permutation(fofmems)
                for dindex in rfofmems:
                    self.data['processed'][dindex] += 1

                    mindex = self.index_list[dindex]
                    print('index: %d:%d' % (mindex,last), )
                    ti = time.time()            
                
                    #skip if flags are set from first try
                    if self.data['flags'][dindex] > 0 and self.data['flags'][dindex] != NO_ATTEMPT:
                        pass
                    else:
                        self.fit_obj(dindex)
                    ti = time.time()-ti
                    print('    time:',ti)
                
                #FIXME: do conv check here
                
                tm=time.time()-t0
                self._try_checkpoint(tm) # only at certain intervals


        #FIXME: do cov comp here...

        tm=time.time()-t0
        print("time:",tm)
        print("time per:",tm/num)
        
        
    def _fit_all_models(self):
        """
        Fit psf flux and other models
        """
        
        if self.data['flags'][dindex] == NO_ATTEMPT:        
            flags=0
            # fit both coadd and se psf flux if exists
            self._fit_psf_flux()

            dindex=self.dindex
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
        if coadd:
            if self.data['processed'][dindex] == -1:
                self.guesser = self._get_guesser('coadd_psf')
            else:
                pars = self.data['coadd_'+model+'_pars_best'][dindex]
                self.guesser = FixedParsGuesser(pars,pars*0.1)
            mb_obs_list=self.sdata['coadd_mb_obs_list']
        else:
            if self.data['processed'][dindex] == -1:
                self.guesser = self._get_guesser('me_psf')
            else:
                pars = self.data[model+'pars_best'][dindex]
                self.guesser = FixedParsGuesser(pars,pars*0.1)
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
        
    def _set_index_list(self):
        """
        set the list of objects to be processed from the fof groups
        """
        if self.fof_range is None:
            start=0
            end=self.nobj_tot-1
            self.index_list = numpy.arange(start,end+1)
        else:
            start=self.fof_range[0]
            end=self.fof_range[1]
            self.index_list = []
            for fofid in range(start,end+1):
                q, = np.where(self.model_data['fofs']['fofid'] == fofid)
                assert len(q) > 0, print 'Found zero length FoF! fofid = %ld' % fofid
                self.index_list.extend(list(self.model_data['fofs']['number'][q]-1))
            self.index_list = np.array(self.index_list)
                
    def _fit_simple_max(self, mb_obs_list, model, nm_pars, ntry=1):
        """
        parameters
        ----------
        mb_obs_list: MultiBandObsList
            The observations to fit
        model: string
            model to fit
        params: dict
            from the config file 'me_iter' or 'coadd_iter'
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
