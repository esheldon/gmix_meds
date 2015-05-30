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
        fracdev_prior = keys.get('fracdev_prior',None)
        fracdev_grid  = keys.get('fracdev_grid',None)
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

    def get_bootstrapper(self, model):
        """
        get the bootstrapper for fitting psf through galaxy
        """
        
        if model=='cm':
            fracdev_prior=self['model_pars']['cm']['fracdev_prior']
            boot=get_bootstrapper(self.sdata['mb_obs_list'],
                                  type='composite',
                                  fracdev_prior=fracdev_prior,
                                  **self)
        else:
            boot=get_bootstrapper(self.sdata['mb_obs_list'], **self)

        return boot

    def _fit_all_models(self):
        """
        Fit psf flux and other models
        """

        flags=0
        for model in self['fit_models']:
            print('    fitting:',model)

            flags |= self._run_fitters(model)

        return flags

    def _run_fitters(self, model):
        from great3.generic import PSFFailure,GalFailure

        flags=0

        dindex=self.dindex
        boot=self.get_bootstrapper(model)

        self.boot=boot

        try:

            # we currently fit the psf elsewhere
            #self._fit_psfs()
            flags |= self._fit_psf_flux()

            if flags == 0:

                try:

                    self._fit_galaxy(model)
                    self._copy_galaxy_result(model)
                    self._print_galaxy_result()

                except BootGalFailure:
                    print("    galaxy fitting failed")
                    flags = GAL_FIT_FAILURE

        except BootPSFFailure:
            print("    psf fitting failed")
            flags = PSF_FIT_FAILURE

        return flags

    def _fit_psf_flux(self):
        self.boot.fit_gal_psf_flux()

        res=self.boot.get_psf_flux_result()

        n=Namer("psf")
        dindex=self.dindex
        data=self.data

        flagsall=0
        for band in xrange(self['nband']):
            flags=res['flags'][band]
            flagsall |= flags

            flux=res['psf_flux'][band]
            flux_err=res['psf_flux_err'][band]

            data[n('flags')][dindex,band] = flags
            data[n('flux')][dindex,band] = flux
            data[n('flux_err')][dindex,band] = flux_err

            print("        psf flux(%s): %g +/- %g" % (band,flux,flux_err))

        return flagsall


    def _fit_psfs(self):
        """
        fit the psf model to every observation's psf image
        """

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

 
    def _fit_max(self, model):
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

        print("        replacing cov")
        boot.try_replace_cov(cov_pars)


        #self.boot.set_round_s2n(max_pars,
        #                        method='sim',
        #                        fitter_type='max')



    def _copy_galaxy_result(self, model):
        """
        Copy from the result dict to the output array
        """

        dindex=self.dindex

        res=self.gal_fitter.get_result()
        mres=self.boot.get_max_fitter().get_result()

        rres=self.boot.get_round_result()

        n=Namer(model)
        data=self.data

        data[n('flags')][dindex] = res['flags']

        fname, Tname = self._get_lnames()

        if res['flags'] == 0:
            pars=res['pars']
            pars_cov=res['pars_cov']

            flux=pars[5:]
            flux_cov=pars_cov[5:, 5:]

            data[n('max_flags')][dindex] = mres['flags']
            data[n('max_pars')][dindex,:] = mres['pars']
            data[n('max_pars_cov')][dindex,:,:] = mres['pars_cov']

            data[n('pars')][dindex,:] = pars
            data[n('pars_cov')][dindex,:,:] = pars_cov

            data[n('g')][dindex,:] = res['g']
            data[n('g_cov')][dindex,:,:] = res['g_cov']

            data[n('flags_r')][dindex]  = rres['flags']
            data[n('s2n_r')][dindex]    = rres['s2n_r']
            data[n(Tname+'_r')][dindex] = rres['pars'][4]
            data[n('T_s2n_r')][dindex]  = rres['T_s2n_r']
            data[n('psf_T_r')][dindex]  = rres['psf_T_r']

            for sn in stat_names:
                data[n(sn)][dindex] = res[sn]

            if self['do_shear']:
                data[n('g_sens')][dindex,:] = res['g_sens']

                if 'R' in res:
                    data[n('P')][dindex] = res['P']
                    data[n('Q')][dindex,:] = res['Q']
                    data[n('R')][dindex,:,:] = res['R']

    def _print_galaxy_result(self):
        res=self.gal_fitter.get_result()

        if 'pars' in res:
            print_pars(res['pars'],    front='    gal_pars: ')
        if 'pars_err' in res:
            print_pars(res['pars_err'],front='    gal_perr: ')


    def _get_lnames(self):
        if self['use_logpars']:
            fname='log_flux'
            Tname='log_T'
        else:
            fname='flux'
            Tname='T'

        return fname, Tname

    def _get_dtype(self):
        self._check_models()

        nband=self['nband']
        bshape=(nband,)
        simple_npars=5+nband

        dt=[('id','i8'),
            ('number','i4'),
            ('processed','i1'),
            ('flags','i4'),
            ('nimage_tot','i4',bshape),
            ('nimage_use','i4',bshape),
            ('time','f8'),

            ('box_size','i2'),

            ('mask_frac','f8'),
            ('psfrec_T','f8'),
            ('psfrec_g','f8', 2)

           ]

        for name in ['psf']:
            n=Namer(name)
            dt += [(n('flags'),   'i4',bshape),
                   (n('flux'),    'f8',bshape),
                   (n('flux_err'),'f8',bshape)]

        if nband==1:
            fcov_shape=(nband,)
        else:
            fcov_shape=(nband,nband)

        
        fname, Tname=self._get_lnames()

        models=self._get_all_models()
        for model in models:

            n=Namer(model)

            np=simple_npars
            
            dt+=[(n('flags'),'i4'),
                 (n('pars'),'f8',np),
                 (n('pars_cov'),'f8',(np,np)),
                 (n('g'),'f8',2),
                 (n('g_cov'),'f8',(2,2)),

                 (n('max_flags'),'i4'),
                 (n('max_pars'),'f8',np),
                 (n('max_pars_cov'),'f8',(np,np)),

                 #(n('max_flags_r'),'i4'),
                 #(n('max_s2n_r'),'f8'),
                 #(n('max_'+Tname+'_r'),'f8'),
                 #(n('max_T_s2n_r'),'f8'),
                
                 (n('s2n_w'),'f8'),
                 (n('chi2per'),'f8'),
                 (n('dof'),'f8'),

                 (n('flags_r'),'i4'),
                 (n('s2n_r'),'f8'),
                 (n(Tname+'_r'),'f8'),
                 (n('T_s2n_r'),'f8'),
                 (n('psf_T_r'),'f8'),
                ]
            
            
            if self['do_shear']:
                dt += [(n('g_sens'), 'f8', 2)]
                       #(n('P'), 'f8'),
                       #(n('Q'), 'f8', 2),
                       #(n('R'), 'f8', (2,2))]
            
        return dt

    def _make_struct(self):
        """
        make the output structure
        """
        dt=self._get_dtype()

        num=self.index_list.size
        data=numpy.zeros(num, dtype=dt)


        data['mask_frac'] = PDEFVAL
        data['psfrec_T'] = DEFVAL
        data['psfrec_g'] = DEFVAL
        
        for name in ['psf']:
            n=Namer(name)
            data[n('flags')] = NO_ATTEMPT
            data[n('flux')] = DEFVAL
            data[n('flux_err')] = PDEFVAL

        fname, Tname=self._get_lnames()

        models=self._get_all_models()
        for model in models:
            n=Namer(model)

            data[n('flags')] = NO_ATTEMPT
            
            data[n('pars')] = DEFVAL
            data[n('pars_cov')] = PDEFVAL*1.e6

            data[n('g')] = DEFVAL
            data[n('g_cov')] = PDEFVAL*1.e6

            data[n('s2n_w')] = DEFVAL
            data[n('chi2per')] = PDEFVAL
            
            data[n('max_flags')] = NO_ATTEMPT
            data[n('max_pars')] = DEFVAL
            data[n('max_pars_cov')] = PDEFVAL*1.e6

            #data[n('max_flags_r')] = NO_ATTEMPT
            #data[n('max_s2n_r')] = DEFVAL
            #data[n('max_'+Tname+'_r')] = DEFVAL
            #data[n('max_T_s2n_r')] = DEFVAL
 
            data[n('flags_r')] = NO_ATTEMPT
            data[n('s2n_r')] = DEFVAL
            data[n(Tname+'_r')] = DEFVAL
            data[n('T_s2n_r')] = DEFVAL
            data[n('psf_T_r')] = DEFVAL
            
            if self['do_shear']:
                data[n('g_sens')] = DEFVAL
                #data[n('P')] = DEFVAL
                #data[n('Q')] = DEFVAL
                #data[n('R')] = DEFVAL

     
        self.data=data


class MedsFitISampleBoot(MedsFitBootBase):
    def _fit_galaxy(self, model):
        self._fit_max(model)
        self._do_isample(model)

        self._add_shear_info(model)

        self.gal_fitter=self.boot.get_isampler()

    def _do_isample(self, model):
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

    def _copy_galaxy_result(self, model):
        super(MedsFitISampleBoot,self)._copy_galaxy_result(model)

        res=self.gal_fitter.get_result()
        if res['flags'] == 0:

            dindex=self.dindex
            res=self.gal_fitter.get_result()
            n=Namer(model)

            for f in ['efficiency','neff']:
                self.data[n(f)][dindex] = res[f]

    def _print_galaxy_result(self):
        super(MedsFitISampleBoot,self)._print_galaxy_result()
        mres=self.boot.get_max_fitter().get_result()

        if 's2n_w' in mres:
            rres=self.boot.get_round_result()
            tup=(mres['s2n_w'],rres['s2n_r'],rres['T_s2n_r'],mres['chi2per'])
            print("    s2n: %.1f s2n_r: %.1f T_s2n_r: %.3g chi2per: %.3f" % tup)


    def _get_dtype(self):

        dt=super(MedsFitISampleBoot,self)._get_dtype()

        for model in self._get_all_models():
            n=Namer(model)
            dt += [
                (n('efficiency'),'f4'),
                (n('neff'),'f4'),
            ]

        return dt

    def _make_struct(self):
        super(MedsFitISampleBoot,self)._make_struct()

        d=self.data
        for model in self._get_all_models():
            n=Namer(model)

            d[n('efficiency')] = DEFVAL
            d[n('neff')] = DEFVAL


class MedsFitISampleBootComposite(MedsFitISampleBoot):

    def _copy_galaxy_result(self, model):
        super(MedsFitISampleBootComposite,self)._copy_galaxy_result(model)

        res=self.gal_fitter.get_result()
        if res['flags'] == 0:

            dindex=self.dindex
            res=self.gal_fitter.get_result()
            n=Namer(model)

            for f in ['fracdev','fracdev_noclip','fracdev_err','TdByTe']:
                self.data[n(f)][dindex] = res[f]

    def _get_dtype(self):

        dt=super(MedsFitISampleBootComposite,self)._get_dtype()

        n=Namer('cm')
        dt += [
            (n('fracdev'),'f4'),
            (n('fracdev_noclip'),'f4'),
            (n('fracdev_err'),'f4'),
            (n('TdByTe'),'f4'),
        ]

        return dt

    def _make_struct(self):
        super(MedsFitISampleBootComposite,self)._make_struct()

        n=Namer('cm')

        d=self.data
        d[n('fracdev')] = PDEFVAL
        d[n('fracdev_noclip')] = PDEFVAL
        d[n('fracdev_err')] = PDEFVAL
        d[n('TdByTe')] = PDEFVAL


