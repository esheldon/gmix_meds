"""
Just do a maxlike fit
"""
from __future__ import print_function, division
from .nfit import *

class MedsFitMax(MedsFit):
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

        if max_s2n >= self['min_psf_s2n']:

            self._do_gaussian_fit()

            for model in self['fit_models']:
                print('    fitting:',model)

                prior=self['model_pars'][model]['prior']
                self._run_model_fit(model, prior)
        else:
            mess="    psf s/n too low: %s (%s)"
            mess=mess % (max_s2n,self['min_psf_s2n'])
            print(mess)
            
            flags |= LOW_PSF_FLUX

        return flags

    def _do_gaussian_fit(self):
        from .util import FromFullParsGuesser
        print('    fitting: gauss')

        mb_obs_list=self.sdata['mb_obs_list']

        self.guesser=self._get_guesser_from_me_psf()

        pmodel=self['fit_models'][0]
        prior=self['model_pars'][pmodel]['prior']

        fitter=self._fit_model(mb_obs_list,"gauss",prior)
        res=fitter.get_result()

        # if OK, guess from this.  Otherwise re-use the psf guesser
        if res['flags']==0:

            self._print_res(fitter)

            widths = res['pars']*0
            widths[0:0+2] = 0.05
            widths[2:2:2] = 0.01
            widths[4] = 0.10
            widths[5:] = 0.05

            self.guesser=FromFullParsGuesser(res['pars'],
                                             res['pars']*0,
                                             widths=widths)
        else:
            print("    gauss fit failed, using psf guesser")


    def _run_model_fit(self, model, prior):
        """
        wrapper to run fit, copy pars, maybe make plots

        sets .fitter or .coadd_fitter
        """

        mb_obs_list=self.sdata['mb_obs_list']

        fitter=self._fit_model(mb_obs_list, model, prior)

        if fitter is not None:
            self._copy_simple_pars(fitter)

            self._print_res(fitter)

            if self['make_plots']:
                self._do_make_plots(fitter, model, coadd=coadd,
                                    fitter_type=self['fitter_class'])

            self.fitter=fitter
        else:
            self.data[n('flags')][self.dindex] = GAL_FIT_FAILURE

    def _fit_model(self, mb_obs_list, model, prior):
        """
        Fit a simple model
        """

        max_pars=self['max_pars']
        method=max_pars['method']

        mess="method should be lm for now"
        assert method=="lm",mess

        fitter=self._fit_simple_lm(mb_obs_list, model, max_pars, prior)
        return fitter

    def _fit_simple_lm(self, mb_obs_list, model, max_pars, prior):
        """
        Fit one of the "simple" models, e.g. exp or dev

        use flat g prior
        """

        from ngmix.fitting import LMSimple


        ntry=max_pars['ntry']
        for i in xrange(ntry):
            guess=self.guesser(prior=prior)

            # catch case where guess is just no good...
            try:
                fitter=LMSimple(mb_obs_list,
                                model,
                                prior=prior,
                                lm_pars=max_pars['lm_pars'])

                fitter.go(guess)
                res=fitter.get_result()
                if res['flags']==0:
                    break
            except GMixRangeError:
                # probably bad guesser
                print("    caught gmix range, using psf guesser")
                self.guesser=self._get_guesser_from_me_psf()
                res={'flags': GAL_FIT_FAILURE}
                fitter=None

        res['ntry']=i+1
        return fitter

    def _copy_simple_pars(self, fitter, coadd=False):
        """
        Copy from the result dict to the output array

        always copy linear result
        """

        dindex=self.dindex
        res=fitter.get_result()

        model=res['model']
        if coadd:
            model = 'coadd_%s' % model

        n=Namer(model)

        self.data[n('flags')][dindex] = res['flags']

        if res['flags'] == 0:
            pars=res['pars']
            pars_cov=res['pars_cov']

            T = pars[4]
            T_s2n = pars[4]/sqrt(pars_cov[4,4])

            flux=pars[5:]
            flux_cov=pars_cov[5:, 5:]

            self.data[n('pars')][dindex,:] = pars
            self.data[n('pars_cov')][dindex,:,:] = pars_cov

            self.data[n('T')][dindex] = pars[4]
            self.data[n('T_s2n')][dindex] = pars[4]

            self.data[n('flux')][dindex] = flux
            self.data[n('flux_cov')][dindex] = flux_cov

            self.data[n('g')][dindex,:] = res['g']
            self.data[n('g_cov')][dindex,:,:] = res['g_cov']

            for sn in stat_names:
                self.data[n(sn)][dindex] = res[sn]

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

            ('coadd_npix','i4'),
            ('coadd_mask_frac','f8'),
            ('coadd_psfrec_T','f8'),
            ('coadd_psfrec_g','f8', 2),

            ('mask_frac','f8'),
            ('psfrec_T','f8'),
            ('psfrec_g','f8', 2)

           ]

        # coadd fit with em 1 gauss
        # the psf flux fits are done for each band separately
        for name in ['coadd_psf','psf']:
            n=Namer(name)
            dt += [(n('flags'),   'i4',bshape),
                   (n('flux'),    'f8',bshape),
                   (n('flux_err'),'f8',bshape),
                   (n('chi2per'),'f8',bshape),
                   (n('dof'),'f8',bshape)]

        if nband==1:
            fcov_shape=(nband,)
        else:
            fcov_shape=(nband,nband)

        models=self._get_all_models()
        for model in models:

            n=Namer(model)

            np=simple_npars
            
            dt+=[(n('flags'),'i4'),
                 (n('pars'),'f8',np),
                 (n('pars_cov'),'f8',(np,np)),
                 (n('T'),'f8'),
                 (n('T_s2n'),'f8'),
                 (n('flux'),'f8',bshape),
                 (n('flux_cov'),'f8',fcov_shape),
                 (n('g'),'f8',2),
                 (n('g_cov'),'f8',(2,2)),
                
                 (n('s2n_w'),'f8'),
                 (n('chi2per'),'f8'),
                 (n('dof'),'f8'),
                ]
            
        return dt

    def _make_struct(self):
        """
        make the output structure
        """
        dt=self._get_dtype()

        num=self.index_list.size
        data=numpy.zeros(num, dtype=dt)

        data['coadd_mask_frac'] = PDEFVAL
        data['coadd_psfrec_T'] = DEFVAL
        data['coadd_psfrec_g'] = DEFVAL

        data['mask_frac'] = PDEFVAL
        data['psfrec_T'] = DEFVAL
        data['psfrec_g'] = DEFVAL
        
        for name in ['coadd_psf','psf']:
            n=Namer(name)
            data[n('flags')] = NO_ATTEMPT
            data[n('flux')] = DEFVAL
            data[n('flux_err')] = PDEFVAL
            data[n('chi2per')] = PDEFVAL

        models=self._get_all_models()
        for model in models:
            n=Namer(model)

            data[n('flags')] = NO_ATTEMPT
            
            data[n('pars')] = DEFVAL
            data[n('pars_cov')] = PDEFVAL*1.e6
            data[n('flux')] = DEFVAL
            data[n('flux_cov')] = PDEFVAL*1.e6
            data[n('g')] = DEFVAL
            data[n('g_cov')] = PDEFVAL*1.e6

            data[n('s2n_w')] = DEFVAL
            data[n('chi2per')] = PDEFVAL
     
        self.data=data



