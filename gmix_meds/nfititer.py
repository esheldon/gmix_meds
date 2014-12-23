from __future__ import print_function
import numpy
from .nfit import *
from ngmix import print_pars

class MHMedsFitHybridIter(MHMedsFitHybrid):
    """
    This version uses MH for fitting, with guess/steps from
    a coadd emcee run, which is seeded via iterating between 
    a direct maximizer and emcee run on the coadd
    """

    def _fit_all_models(self):
        """
        Fit psf flux and other models
        """

        flags=0
        # fit both coadd and se psf flux if exists
        self._fit_psf_flux()

        dindex=self.dindex
        s2n=self.data['psf_flux'][dindex,:]/self.data['psf_flux_err'][dindex,:]
        max_s2n=numpy.nanmax(s2n)
        
        if max_s2n >= self['min_psf_s2n'] and len(self['fit_models']) > 0:
            for model in self['fit_models']:
                print('    fitting:',model)
                
                if self['fit_coadd_galaxy']:
                    print('    coadd iter fit')                
                    self.coadd_guesser = \
                        self._guess_params_iter(self.sdata['coadd_mb_obs_list'], 
                                                model, 
                                                self['coadd_iter'],
                                                self._get_guesser('coadd_psf'))
                    if self.coadd_guesser == None:
                        self.coadd_guesser = self._get_guesser('coadd_psf')
                    
                    print('    coadd')                
                    self._run_model_fit(model, self['coadd_fitter_class'],coadd=True)

                if self['fit_me_galaxy']:
                    if 'me_iter' in self:
                        print('    multi-epoch iter fit')                
                        self.me_guesser = \
                            self._guess_params_iter(self.sdata['mb_obs_list'],
                                                    model, 
                                                    self['me_iter'], 
                                                    self._get_guesser('me_psf'))
                    else:
                        self.me_guesser = None
                    if self.me_guesser == None:
                        self.me_guesser = self._get_guesser('me_psf')
                        
                    print('    multi-epoch')
                    # fitter class should be mh...
                    self._run_model_fit(model, self['fitter_class'], coadd=False)
        else:
            mess="    psf s/n too low: %s (%s)"
            mess=mess % (max_s2n,self['min_psf_s2n'])
            print(mess)
            
            flags |= LOW_PSF_FLUX

        return flags

    def _guess_params_iter(self, mb_obs_list, model, params, start):
        fmt = "%10.6g "*(5+self['nband'])

        skip_emcee = params.get('skip_emcee',False)
        skip_nm = params.get('skip_nm',False)

        print("        doing iterative init")

        for i in xrange(params['max']):
            print('        iter % 3d of %d' % (i+1,params['max']))

            if i == 0:
                self.guesser = start

            if not skip_emcee:
                emceefit = self._fit_simple_emcee_guess(mb_obs_list, model, params)

                emcee_pars = emceefit.get_best_pars()
                bestlk = emceefit.get_best_lnprob()
                if self['print_params']:
                    print('            emcee max: ',
                          fmt % tuple(emcee_pars), 'logl:   %lf' % bestlk)
                else:
                    print('            emcee max loglike: %lf' % bestlk)
                
                # making up errors, but it doesn't matter                    
                self.guesser = FixedParsGuesser(emcee_pars,emcee_pars*0.1)

            if not skip_nm:
                # first nelder mead
                greedyfit1 = self._fit_simple_max(mb_obs_list, model, params)
                res1=greedyfit1.get_result()
                
                bestlk = greedyfit1.calc_lnprob(res1['pars'])
                if self['print_params']:
                    print('            nm max:    ',
                          fmt % tuple(res1['pars']),'logl:    %lf' % bestlk)
                else:
                    print('            nm max loglike: %lf' % bestlk)
            
                # must ignore errors in nedler-mead
                doemcee=True
                self.guesser = FromFullParsGuesser(res1['pars'],res1['pars']*0.1)

            greedyfit2 = self._fit_simple_lm(mb_obs_list, model, params)
            res2=greedyfit2.get_result()

            if res2['flags'] == 0:
                pars_check=numpy.all(numpy.abs(res2['pars']) < 1e9)
                if pars_check:
                    pars=res2['pars']
                    #self.guesser=FixedParsGuesser(pars,res2['pars_err'])
                    self.guesser=FixedParsCovGuesser(pars,res2['pars_cov'])
                    doemcee=False

                    bestlk = greedyfit2.calc_lnprob(pars)
                    if self['print_params']:
                        print('            lm max:    ',
                              fmt % tuple(pars),'logl:    %lf' % bestlk)
                    else:
                        print('            lm max loglike: %lf' % bestlk)
                    print("            nfev:",res2['nfev'])

                else:
                    print("bad lm pars")
            else:
                print("lm failed")

            if doemcee:
                print("        greedy failure, running emcee")
                # something went wrong, so just continue emcee
                # where we left off.  is 400 about right?
                pos=emceefit.get_last_pos()
                emceefit.run_mcmc(pos,400)
                emceefit.calc_result()
                res=emceefit.get_result()
                pars=emceefit.get_best_pars()
                #self.guesser=FixedParsGuesser(pars,res['pars_err'])
                self.guesser=FixedParsCovGuesser(pars,res['pars_cov'])

                bestlk = emceefit.get_best_lnprob()
                if self['print_params']:
                    print('            emcee2 min:',
                          fmt % tuple(pars),'logl:    %lf' % bestlk)
                else:
                    print('            emcee2 min loglike: %lf' % bestlk)
        
        return self.guesser

    def _run_model_fit(self, model, fitter_type, coadd=False):
        """
        wrapper to run fit, copy pars, maybe make plots

        sets .fitter or .coadd_fitter

        this one does not currently use self['guess_type']
        """

        if coadd:
            self.guesser=self.coadd_guesser
            mb_obs_list=self.sdata['coadd_mb_obs_list']
        else:
            self.guesser=self.me_guesser
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

    def _fit_simple_emcee_guess(self, mb_obs_list, model, params):
        """
        Fit one of the "simple" models, e.g. exp or dev

        use flat g prior
        """

        from ngmix.fitting import MCMCSimple

        # note flat on g!
        prior=self['model_pars'][model]['gflat_prior']

        epars=params['emcee_pars']
        guess=self.guesser(n=epars['nwalkers'], prior=prior)

        fitter=MCMCSimple(mb_obs_list,
                          model,
                          nu=self['nu'],
                          margsky=self['margsky'],
                          prior=prior,
                          nwalkers=epars['nwalkers'],
                          mca_a=epars['a'],
                          random_state=self.random_state)

        pos=fitter.run_mcmc(guess,epars['burnin'])
        pos=fitter.run_mcmc(pos,epars['nstep'])

        return fitter

    def _fit_simple_lm(self, mb_obs_list, model, params, use_prior=True):
        """
        Fit one of the "simple" models, e.g. exp or dev

        use flat g prior
        """

        from ngmix.fitting import LMSimple

        if use_prior:
            prior=self['model_pars'][model]['gflat_prior']
        else:
            prior=None

        ntry=params['lm_ntry']
        for i in xrange(ntry):
            guess=self.guesser(prior=prior)

            fitter=LMSimple(mb_obs_list,
                            model,
                            prior=prior,
                            lm_pars=params['lm_pars'])

            fitter.run_lm(guess)
            res=fitter.get_result()
            if res['flags']==0:
                break

        res['ntry']=i+1
        return fitter

    def _fit_simple_max(self, mb_obs_list, model, params):
        from ngmix.fitting import MaxSimple        

        prior=self['model_pars'][model]['gflat_prior']

        guess=self.guesser(prior=prior)
        fitter=MaxSimple(mb_obs_list,
                         model,
                         prior=prior,
                         method=params['min_method'])
        fitter.run_max(guess)
        return fitter
                            

class MedsFitEmceeIter(MedsFit):
    """
    Guesses from a greeder optimizer, emcee for the chain
    """

    def _fit_all_models(self):
        """
        Fit psf flux and other models
        """

        flags=0
        # fit both coadd and se psf flux if exists
        self._fit_psf_flux()

        dindex=self.dindex
        s2n=self.data['psf_flux'][dindex,:]/self.data['psf_flux_err'][dindex,:]
        max_s2n=numpy.nanmax(s2n)
        
        if max_s2n >= self['min_psf_s2n'] and len(self['fit_models']) > 0:
            for model in self['fit_models']:
                print('    fitting:',model)

                # sets self.guesser
                self._guess_params_iter(self.sdata['mb_obs_list'],
                                        model, 
                                        self['me_iter'], 
                                        self._get_guesser('me_psf'))
                
                self._run_model_fit(model)
        else:
            mess="    psf s/n too low: %s (%s)"
            mess=mess % (max_s2n,self['min_psf_s2n'])
            print(mess)
            
            flags |= LOW_PSF_FLUX

        return flags

    def _run_model_fit(self, model):
        """
        wrapper to run fit, copy pars, maybe make plots

        sets .fitter
        """

        mb_obs_list=self.sdata['mb_obs_list']

        fitter=self._fit_model(mb_obs_list, model)

        self._copy_simple_pars(fitter)

        self._print_res(fitter)

        if self['make_plots']:
            self._do_make_plots(fitter, model)

        self.fitter=fitter


    def _guess_params_iter(self, mb_obs_list, model, params, start):
        fmt = "%10.6g "*(5+self['nband'])

        print("        using method '%s' for minimizer" % params['min_method'])

        for i in xrange(params['max']):
            print('        iter % 3d of %d' % (i+1,params['max']))

            if i == 0:
                self.guesser = start

            emceefit = self._fit_simple_emcee_guess(mb_obs_list, model, params)

            emcee_pars = emceefit.get_best_pars()
            bestlk = emceefit.get_best_lnprob()
            print('            emcee min: ',
                  fmt % tuple(emcee_pars), 'loglike = %lf' % bestlk)

            # making up errors, but it doesn't matter                    
            self.guesser = FixedParsGuesser(emcee_pars,emcee_pars*0.1)

            # first nelder mead
            greedyfit = self._fit_simple_max(mb_obs_list, model, params)
            res=greedyfit.get_result()

            bestlk = greedyfit.calc_lnprob(res['pars'])
            print('            nm min:    ',
                  fmt % tuple(res['pars']),'loglike = %lf' % bestlk)

            self.guesser = FromFullParsGuesser(res['pars'],emcee_pars*0.1)

    def _fit_simple_emcee_guess(self, mb_obs_list, model, params):
        """
        Fit one of the "simple" models, e.g. exp or dev

        use flat g prior
        """

        from ngmix.fitting import MCMCSimple

        # note flat on g!
        prior=self['model_pars'][model]['gflat_prior']

        epars=params['emcee_pars']
        guess=self.guesser(n=epars['nwalkers'], prior=prior)

        fitter=MCMCSimple(mb_obs_list,
                          model,
                          nu=self['nu'],
                          margsky=self['margsky'],
                          prior=prior,
                          nwalkers=epars['nwalkers'],
                          mca_a=epars['a'],
                          random_state=self.random_state)

        pos=fitter.run_mcmc(guess,epars['burnin'])
        pos=fitter.run_mcmc(pos,epars['nstep'])

        return fitter

    def _fit_simple_max(self, mb_obs_list, model, params):
        from ngmix.fitting import MaxSimple        

        prior=self['model_pars'][model]['gflat_prior']

        guess=self.guesser(prior=prior)
        fitter=MaxSimple(mb_obs_list,
                         model,
                         prior=prior,
                         method=params['min_method'])
        fitter.run_max(guess)
        return fitter
 

