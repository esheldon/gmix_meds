from __future__ import print_function
import numpy
from .nfit import *
from ngmix import print_pars

from ngmix.fitting import MaxSimple, EIG_NOTFINITE

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
                    mb_obs_list=self.sdata['mb_obs_list']
                    if 'me_iter' in self:
                        print('    multi-epoch iter fit')                
                        self.me_guesser = \
                            self._guess_params_iter(mb_obs_list,
                                                    model, 
                                                    self['me_iter'], 
                                                    self._get_guesser('me_psf'))
                    else:
                        self.me_guesser = None
                    if self.me_guesser == None:
                        self.me_guesser = self._get_guesser('me_psf')
                        
                    if self['use_guess_aper']:
                        self._set_aperture_from_pars(mb_obs_list,
                                                     self.me_guesser.pars.copy())

                    print('    multi-epoch')
                    # fitter class should be mh...
                    self._run_model_fit(model, self['fitter_class'], coadd=False)
        else:
            mess="    psf s/n too low: %s (%s)"
            mess=mess % (max_s2n,self['min_psf_s2n'])
            print(mess)
            
            flags |= LOW_PSF_FLUX

        return flags

    def _guess_params_iter_test(self, mb_obs_list, model, params, start):
        self.fmt = "%10.6g "*(5+self['nband'])

        print("        doing iterative init")

        maxiter=params['maxiter']
        for i in xrange(maxiter):
            print('        iter % 3d of %d' % (i+1,maxiter))

            if i == 0:
                self.guesser = start

            epars=params['emcee_pars']
            self._do_emcee_guess(mb_obs_list,
                                 model,
                                 epars,
                                 guesser2make='fixed')

            self._do_nm_guess_one(mb_obs_list,
                                  model,
                                  params['nm_pars'])

            self._do_emcee_guess(mb_obs_list,
                                 model,
                                 params.get('emcee_pars2',epars),
                                 guesser2make='fixed-cov')
        
        return self.guesser

    def _guess_params_iter(self, mb_obs_list, model, params, start):
        mhpars=self['mh_pars']
        fac=mhpars['step_factor']

        self.fmt = "%10.6g "*(5+self['nband'])

        emcee_only = params.get('emcee_only',False)

        print("        doing iterative init")

        maxiter=params['maxiter']
        for i in xrange(maxiter):
            print('        iter % 3d of %d' % (i+1,maxiter))

            if i == 0:
                self.guesser = start

            if emcee_only:
                self._do_emcee_guess(mb_obs_list,
                                     model,
                                     params['emcee_long_pars'],
                                     guesser2make='fixed-cov')
            else:
                epars=params['emcee_quick_pars']
                self._do_emcee_guess(mb_obs_list,
                                     model,
                                     epars)

                res, ok = self._do_nm_guess(mb_obs_list,
                                            model,
                                            params['nm_pars'])
                if ok:
                    # check step sizes
                    tsteps=fac*res['pars_err']
                    wbad,=numpy.where(  (tsteps > self.max_steps_dict[model])
                                      | (tsteps < self.min_steps) )
                    if wbad.size > 0:
                        print("        errors out of bounds")
                        ok=False
                if not ok:
                    print("        greedy failure, running emcee")
                    self.guesser=FromParsGuesser(res['pars'],res['pars']*0.1)
                    self._do_emcee_guess(mb_obs_list,
                                         model,
                                         params.get('emcee_long_pars',epars),
                                         guesser2make='fixed-cov')
            
        return self.guesser


    def _do_emcee_guess(self, mb_obs_list, model, epars, guesser2make='fixed'):

        print('        emcee for guess')

        fitter = self._fit_simple_emcee_guess(mb_obs_list,
                                              model,
                                              epars)

        # might have crazy values, and this is just for a guess anyway
        fitter.calc_result(sigma_clip=True, niter=10)

        res=fitter.get_result()

        best_pars = fitter.get_best_pars()
        self.bestlk = fitter.get_best_lnprob()
        arate=fitter.get_arate()

        print('            emcee arate:',arate,'logl:', self.bestlk)
        if self['print_params']:
            print_pars(best_pars, front='        ')
            print_pars(res['pars_err'],front='        ')
        
        # making up errors, but it doesn't matter

        if guesser2make=='fixed':
            self.guesser = FixedParsGuesser(best_pars,best_pars*0.1)
        elif guesser2make=='fixed-cov':
            self.guesser = FixedParsCovGuesser(best_pars,res['pars_cov'])
        else:
            self.guesser = FromParsGuesser(best_pars,best_pars*0.1)

        self.emceefit=fitter

    def _do_nm_guess_one(self, mb_obs_list, model, nm_pars):
        print('        nm for guess')

        ntry=1
        fitter = self._fit_simple_max(mb_obs_list,
                                      model,
                                      nm_pars,
                                      ntry=ntry)
        res=fitter.get_result()
        
        pars=res['pars']
        self.bestlk = fitter.calc_lnprob(pars)

        if self['print_params']:
            print('            nm max:    ',
                  self.fmt % tuple(pars),'logl:    %lf' % self.bestlk)
            if 'pars_err' in res:
                print('            nm err:    ',
                      self.fmt % tuple(res['pars_err']))
        else:
            print('            nm max loglike: %lf' % self.bestlk)
    
        self.guesser=FromParsGuesser(pars,pars*0.1,widths=pars*0+0.01)
        self.greedyfit = fitter
        return res


    def _do_nm_guess(self, mb_obs_list, model, nm_pars):
        """
        this is only for a guess, so we only care if the covariance
        matrix is OK
        """

        print('        nm for guess')

        ntry=nm_pars['ntry']
        fitter = self._fit_simple_max(mb_obs_list,
                                      model,
                                      nm_pars,
                                      ntry=ntry)


        res=fitter.get_result()
        if (res['flags'] & EIG_NOTFINITE) != 0:
            ok=False
        else:
            ok=True
        
        pars=res['pars']
        self.bestlk = fitter.calc_lnprob(pars)

        if self['print_params']:
            print('            nm max:    ',
                  self.fmt % tuple(pars),'logl:    %lf' % self.bestlk)
            if 'pars_err' in res:
                print('            nm err:    ',
                      self.fmt % tuple(res['pars_err']))
            else:
                print('            no errors found')
    
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
        print_pars(guess.mean(axis=0), front="            emcee guess:")

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
