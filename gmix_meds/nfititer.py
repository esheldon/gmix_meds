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
        self.fmt = "%10.6g "*(5+self['nband'])

        skip_emcee = params.get('skip_emcee',False)
        skip_nm = params.get('skip_nm',False)
        cov_from_lm = params.get('cov_from_lm',False)

        print("        doing iterative init")

        niter=params['niter']
        for i in xrange(niter):
            print('        iter % 3d of %d' % (i+1,niter))

            if i == 0:
                self.guesser = start

            if not skip_emcee:
                self._do_emcee_guess(mb_obs_list,
                                     model,
                                     params['emcee_pars'])

            res, ok = self._do_nm_guess(mb_obs_list,
                                        model,
                                        params['nm_pars'])

            if not ok:
                print("        greedy failure, running emcee")

                self.guesser=FixedParsGuesser(res['pars'],res['pars']*0.1)
                epars={'nwalkers':20,'burnin':200,'nstep':200}
                self._do_emcee_guess(mb_obs_list,
                                     model,
                                     epars,
                                     make_cov_guesser=True)
        
        return self.guesser

    def _do_emcee_guess(self, mb_obs_list, model, epars, make_cov_guesser=False):
        print('        fitting emcee for guess')
        fitter = self._fit_simple_emcee_guess(mb_obs_list,
                                                model,
                                                epars)
        pars = fitter.get_best_pars()
        self.bestlk = fitter.get_best_lnprob()
        if self['print_params']:
            print('            emcee max: ',
                  self.fmt % tuple(pars), 'logl:   %lf' % self.bestlk)
        else:
            print('            emcee max loglike: %lf' % self.bestlk)
        
        # making up errors, but it doesn't matter
        if make_cov_guesser:
            self.guesser=FixedParsCovGuesser(pars,res['pars_cov'])
        else:
            self.guesser = FixedParsGuesser(pars,pars*0.1)

        self.emceefit=fitter

    def _do_nm_guess(self, mb_obs_list, model, nm_pars):
        print('        fitting nm for guess')
        fitter, ok = self._fit_simple_max(mb_obs_list,
                                          model,
                                          nm_pars)
        res=fitter.get_result()
        
        pars=res['pars']
        self.bestlk = fitter.calc_lnprob(pars)

        if self['print_params']:
            print('            nm max:    ',
                  self.fmt % tuple(pars),'logl:    %lf' % self.bestlk)
        else:
            print('            nm max loglike: %lf' % self.bestlk)
    
        self.greedyfit = fitter

        if ok:
            # one more check
            ok=numpy.all(numpy.abs(res['pars']) < 1e10)
            if ok:
                self.guesser=FixedParsCovGuesser(pars,res['pars_cov'])
        return res, ok

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

    def _fit_simple_emcee_guess(self, mb_obs_list, model, epars):
        """
        Fit one of the "simple" models, e.g. exp or dev

        use flat g prior
        """

        from ngmix.fitting import MCMCSimple

        # note flat on g!
        prior=self['model_pars'][model]['gflat_prior']

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

        ok=False
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
                ok=True
                break

        res['ntry']=i+1
        return fitter

    def _fit_simple_max(self, mb_obs_list, model, nm_pars, flags2check='cov'):
        """
        parameters
        ----------
        mb_obs_list: MultiBandObsList
            The observations to fit
        model: string
            model to fit
        params: dict
            from the config file 'me_iter' or 'coadd_iter'
        flags2check: string
            if 'cov' only check EIG_NOTFINITE which indicates if the covariance
            matrix was OK; this is fine when the result is only used as a guess
            and full convergence is not required.  Otherwise check all flags
            equal zero.
        """
        from ngmix.fitting import MaxSimple, EIG_NOTFINITE

        prior=self['model_pars'][model]['gflat_prior']

        method = 'Nelder-Mead'

        fitter=MaxSimple(mb_obs_list,
                         model,
                         prior=prior,
                         method=method)

        ok=False
        for i in xrange(nm_pars['ntry']):
            guess=self.guesser(prior=prior)
            fitter.run_max(guess, **nm_pars)
            res=fitter.get_result()

            if flags2check=='cov':
                if (res['flags'] & EIG_NOTFINITE) == 0:
                    ok=True
                    break
                print("        bad cov, retrying nm")
            else:
                if res['flags']==0:
                    ok=True
                    break
                print("        fit failure, retrying nm")

        return fitter, ok
