"""
use an ngmix bootstrapper to fit

currently the g prior is always applied for max like
and isample
"""
from __future__ import print_function
import numpy
from .nfit import *
from ngmix import print_pars

from ngmix.fitting import EIG_NOTFINITE
from ngmix.gexceptions import BootPSFFailure,BootGalFailure

def get_bootstrapper(obs, type='boot', **keys):
    from ngmix.bootstrap import Bootstrapper
    from ngmix.bootstrap import CompositeBootstrapper
    from ngmix.bootstrap import BestBootstrapper

    use_logpars=keys.get('use_logpars',True)

    if type=='boot':
        #print("    loading bootstrapper")
        boot=Bootstrapper(obs,
                          use_logpars=use_logpars)
    elif type=='composite':
        #print("    loading composite bootstrapper")
        fracdev_prior = keys['fracdev_prior']
        fracdev_grid  = keys['fracdev_grid']
        boot=CompositeBootstrapper(obs,
                                   fracdev_prior=fracdev_prior,
                                   fracdev_grid=fracdev_grid,
                                   use_logpars=use_logpars)
    else:
        raise ValueError("bad bootstrapper type: '%s'" % type)

    return boot

class MedsFitBootBase(MedsFit):
    """
    Use a ngmix bootstrapper
    """

    def get_bootstrapper(self):
        """
        get the bootstrapper for fitting psf through galaxy
        """
        
        if model=='cm':
            boot=get_bootstrapper(self.sdata['mb_obs_list'],
                                  type='composite',
                                  **self)
        else:
            boot=get_bootstrapper(self.sdata['mb_obs_list'], **self)

        return boot

    def _fit_all_models(self):
        """
        Fit psf flux and other models
        """

        for model in self['fit_models']:
            print('    fitting:',model)

            self._run_fitters(model)


    def _run_fitters(self, model):
        from great3.generic import PSFFailure,GalFailure

        flags=0

        dindex=self.dindex
        boot=self.get_bootstrapper(model)

        self.boot=boot

        try:

            self._fit_psfs()
            self._fit_psf_flux()

            try:

                self._fit_galaxy(model)
                self._copy_galaxy_result()
                self._print_galaxy_result()

            except BootGalFailure:
                print("    galaxy fitting failed")
                flags = GAL_FIT_FAILURE

        except BootPSFFailure:
            print("    psf fitting failed")
            flags = PSF_FIT_FAILURE


    def _fit_psfs(self):

        dindex=self.dindex
        boot=self.boot

        psf_pars=self['psf_pars']

        boot.fit_psfs(psf_pars['model'],
                      Tguess=self['psf_Tguess'],
                      ntry=psf_pars['ntry'])

        self.copy_psf_result()

    def fit_psf_flux(self):
        """
        fit psf model to galaxy with one free parameter for flux
        """
        boot=self.boot
        dindex=self.dindex

        boot.fit_gal_psf_flux()

        data=self.data

        pres = boot.get_psf_flux_result()
        data['psf_flux'][dindex] = pres['psf_flux'][0]
        data['psf_flux_err'][dindex] = pres['psf_flux_err'][0]

    def _fit_galaxy(self, model):
        """
        over-ride for different fitters
        """
        raise RuntimeError("over-ride me")

 
    def fit_max(self, model):
        """
        do a maximum likelihood fit

        note prior applied during
        """
        boot=self.boot

        max_pars=self['max_pars']

        cov_pars=self['cov_pars']

        prior=self['model_pars'][model]['prior']

        # now with prior
        print("fitting with g prior")
        boot.fit_max(model,
                     max_pars,
                     prior=prior,
                     ntry=max_pars['ntry'])
        boot.try_replace_cov(cov_pars)


        self.boot.set_round_s2n(max_pars,
                                method='sim',
                                fitter_type='max')


class MedsFitISampleBoot(MedsFitBootBase):
    def _fit_galaxy(self, model):
        self._fit_max(model)
        self._do_isample()

        self._add_shear_info(model)

        self.fitter=self.boot.get_isampler()

    def do_isample(self):
        """
        run isample on the bootstrapper
        """
        ipars=self['isample_pars']
        prior=self['model_pars'][model]['prior']
        self.boot.isample(ipars, prior=prior)

        self.boot.set_round_s2n(self['max_pars'],
                                method='sim',
                                fitter_type='isample')


    def _add_shear_info(self, model):
        """
        add shear information based on the gal_fitter
        """

        boot=self.boot
        max_fitter=boot.get_max_fitter()
        sampler=boot.get_isampler()

        # this is the full prior
        prior=self['model_pars'][model]['prior']
        g_prior=prior.g_prior

        iweights = sampler.get_iweights()
        samples = sampler.get_samples()
        g_vals=samples[:,2:2+2]

        res=sampler.get_result()

        # keep for later if we want to make plots
        self.weights=iweights

        # we are going to mutate the result dict owned by the sampler
        stats = max_fitter.get_fit_stats(res['pars'])
        res.update(stats)

        ls=ngmix.lensfit.LensfitSensitivity(g_vals,
                                            g_prior,
                                            weights=iweights,
                                            remove_prior=True)
        g_sens = ls.get_g_sens()
        g_mean = ls.get_g_mean()

        res['g_sens'] = g_sens
        res['nuse'] = ls.get_nuse()

 

class MedsFitISampleBootComposite(MedsFitBootBase):
    pass

