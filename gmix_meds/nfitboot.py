"""
use an ngmix bootstrapper to fit
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
    elif type=='best': 
        #print("    loading best bootstrapper")
        boot=BestBootstrapper(obs,
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
        
        raise RuntimeError("over-ride")
        return boot

    def _run_fitters(self):
        from great3.generic import PSFFailure,GalFailure

        flags=0

        dindex=self.dindex
        boot=self.get_bootstrapper()

        self.boot=boot

        try:

            self._fit_psfs()
            self._fit_psf_flux()

            try:

                self._fit_galaxy()
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

        #self.psf_fitter=self.boot.get_psf_fitter()
        # for multi-obs this will be the latest
        self.psf_fitter=self.boot.psf_fitter

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

    def _fit_galaxy(self):
        """
        over-ride for different fitters
        """
        raise RuntimeError("over-ride me")

 


class MedsFitISampleBootComposite(MedsFitBootBase):
    def get_bootstrapper(self):
        """
        get the bootstrapper for fitting psf through galaxy
        """
        
        return get_bootstrapper(mb_obs,
                                type='composite',
                                **self)

    def _fit_galaxy(self):
        self._fit_max()
        self._do_isample()

        self._add_shear_info()

        self.fitter=self.boot.get_isampler()


