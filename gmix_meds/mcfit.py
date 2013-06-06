from .lmfit import MedsFit, get_model_names, \
        get_psf_ngauss, add_noise_matched, sigma_clip, \
        _stat_names

class MedsMCMC(MedsFit):
    def __init__(self, meds_file, **keys):

        super(MedsMCMC,self).__init__(self, meds_file, gprior, **keys)

        self.gprior=gprior
        self.nwalkers=keys.get('nwalkers',20)
        self.burnin=keys.get('burnin',400)
        self.nstep=keys.get('nstep',200)
        self.do_pqr=keys.get("do_pqr",False)
        self.mca_a=keys.get('mca_a',2.0)
        self.cen_width = 0.27 # ''

    def fit_obj(self, index):
        """
        Process the indicated object

        The first cutout is always the coadd, followed by
        the SE images which will be fit simultaneously
        """

        t0=time.time()
        if self.meds['ncutout'][index] < 2:
            self.data['flags'][index] |= NO_SE_CUTOUTS
            return

        imlist0=self._get_imlist(index)
        wtlist0=self._get_wtlist(index)
        jacob_list0=self._get_jacobian_list(index)
        self.data['nimage_tot'][index] = len(imlist0)
    
        keep_list,psf_gmix_list=self._fit_psfs(index,jacob_list0)
        if len(psf_gmix_list)==0:
            self.data['flags'][index] |= PSF_FIT_FAILURE
            return

        keep_list,psf_gmix_list=self._remove_bad_psfs(keep_list,psf_gmix_list)
        if len(psf_gmix_list)==0:
            self.data['flags'][index] |= PSF_LARGE_OFFSETS
            return

        imlist = [imlist0[i] for i in keep_list]
        wtlist = [wtlist0[i] for i in keep_list]
        jacob_list = [jacob_list0[i] for i in keep_list]
        self.data['nimage_use'][index] = len(imlist)

        sdata={'imlist':imlist,
               'wtlist':wtlist,
               'jacob_list':jacob_list,
               'psf_gmix_list':psf_gmix_list}

        self._fit_psf_flux(index, sdata)
        self._fit_simple_models(index, sdata)
        self._fit_cmodel(index, sdata)
        self._fit_match(index, sdata)

        if self.debug >= 3:
            self._debug_image(sdata['imlist'][0],sdata['wtlist'][-1])

        self.data['time'][index] = time.time()-t0

    def _fit_simple_models(self, index, sdata):
        """
        Fit all the simple models
        """
        if self.debug:
            bsize=self.meds['box_size'][index]
            bstr='[%d,%d]' % (bsize,bsize)
            print >>stderr,'\tfitting simple models %s' % bstr

        for model in self.simple_models:
            gm=self._fit_simple(model, sdata)
            res=gm.get_result()

            n=get_model_names(model)

            if self.debug:
                self._print_simple_stats(n, res)

            self._copy_simple_pars(index, res, n)

    def _fit_simple(self, model, sdata):
        """
        Fit one of the "simple" models, e.g. exp or dev
        """
        gm=MixMCSimple(sdata['imlist'],
                       sdata['wtlist'],
                       sdata['psf_gmix_list'],
                       self.gprior,
                       T_guess,
                       cen_guess,
                       model,
                       jacob=sdata['jacob_list'],
                       cen_width=self.cen_width,
                       nwalkers=self.nwalkers,
                       burnin=self.burnin,
                       nstep=self.nstep)
        return gm

    def _copy_simple_pars(self, index, res, n):

        self.data[n['flags']][index] = res['flags']

        if res['flags'] == 0:
            self.data[n['pars']][index,:] = res['pars']
            self.data[n['pars_cov']][index,:,:] = res['pcov']

            flux=res['pars'][5]
            flux_err=sqrt(res['pcov'][5,5])
            self.data[n['flux']][index] = flux
            self.data[n['flux_err']][index] = flux_err

            self.data[n['g']][index,:] = res['pars'][2:2+2]
            self.data[n['g_cov']][index,:,:] = res['pcov'][2:2+2,2:2+2]

            self.data[n['g_sens']][index,:] = res['gsens']
            if self.do_pqr:
                self.data[n['P']][index] = res['P']
                self.data[n['Q']][index,:] = res['Q']
                self.data[n['R']][index,:,:] = res['R']

            for sn in _stat_names:
                self.data[n[sn]][index] = res[sn]
        else:
            if self.debug:
                print >>stderr,'flags != 0, errmsg:',res['errmsg']
            if self.debug > 1 and self.debug < 3:
                self._debug_image(sdata['imlist'][0],sdata['wtlist'][0])

    def _print_simple_stats(self, ndict, res):                        
        fmt='\t\t%s: %g +/- %g'
        n=ndict
        if res['flags']==0:
            nm=n['flux']
            flux=res['pars'][5]
            flux_err=sqrt(res['pcov'][5,5])
            print >>stderr,fmt % (nm,flux,flux_err)



    def _make_struct(self):
        nobj=self.meds.size

        dt=[('id','i4'),
            ('flags','i4'),
            ('nimage_tot','i4'),
            ('nimage_use','i4'),
            ('time','f8')]

        simple_npars=6
        simple_models=self.simple_models
        for model in simple_models:
            n=get_model_names(model)

            dt+=[(n['flags'],'i4'),
                 (n['pars'],'f8',simple_npars),
                 (n['pars_cov'],'f8',(simple_npars,simple_npars)),
                 (n['flux'],'f8'),
                 (n['flux_err'],'f8'),
                 (n['g'],'f8',2),
                 (n['g_cov'],'f8',(2,2)),
                 (n['g_sens'],'f8',2),
                 (n['P'],'f8'),
                 (n['Q'],'f8',2),
                 (n['R'],'f8',(2,2)),
                
                 (n['s2n_w'],'f8'),
                 (n['loglike'],'f8'),
                 (n['chi2per'],'f8'),
                 (n['dof'],'f8'),
                 (n['fit_prob'],'f8'),
                 (n['aic'],'f8'),
                 (n['bic'],'f8'),
                ]

        dt += [('cmodel_flags','i4'),
               ('cmodel_flux','f8'),
               ('cmodel_flux_err','f8'),
               ('frac_dev','f8'),
               ('frac_dev_err','f8')]

        n=get_model_names('psf')
        dt += [('psf_flags','i4'),
               ('psf_pars','f8',3),
               ('psf_pars_cov','f8',(3,3)),
               ('psf_flux','f8'),
               ('psf_flux_err','f8'),
               (n['s2n_w'],'f8'),
               (n['loglike'],'f8'),
               (n['chi2per'],'f8'),
               (n['dof'],'f8'),
               (n['fit_prob'],'f8'),
               (n['aic'],'f8'),
               (n['bic'],'f8')]

        dt +=[('match_flags','i4'),
              ('match_model','S3'),
              ('match_flux','f8'),
              ('match_flux_err','f8'),
              ]


        data=numpy.zeros(nobj, dtype=dt)
        data['id'] = 1+numpy.arange(nobj)

        data['cmodel_flags'] = NO_ATTEMPT
        data['cmodel_flux'] = DEFVAL
        data['cmodel_flux_err'] = PDEFVAL
        data['frac_dev'] = DEFVAL
        data['frac_dev_err'] = PDEFVAL

        data['psf_flags'] = NO_ATTEMPT
        data['psf_pars'] = DEFVAL
        data['psf_pars_cov'] = PDEFVAL
        data['psf_flux'] = DEFVAL
        data['psf_flux_err'] = PDEFVAL

        data['psf_s2n_w'] = DEFVAL
        data['psf_loglike'] = BIG_DEFVAL
        data['psf_chi2per'] = PDEFVAL
        data['psf_aic'] = BIG_PDEFVAL
        data['psf_bic'] = BIG_PDEFVAL


        data['match_flags'] = NO_ATTEMPT
        data['match_flux'] = DEFVAL
        data['match_flux_err'] = PDEFVAL
        data['match_model'] = 'nil'


        for model in simple_models:
            n=get_model_names(model)

            data[n['rfc_flags']] = NO_ATTEMPT
            data[n['flags']] = NO_ATTEMPT

            data[n['rfc_pars']] = DEFVAL
            data[n['rfc_pars_cov']] = PDEFVAL
            data[n['pars']] = DEFVAL
            data[n['pars_cov']] = PDEFVAL
            data[n['flux']] = DEFVAL
            data[n['flux_err']] = PDEFVAL
            data[n['g']] = DEFVAL
            data[n['g_cov']] = PDEFVAL
            data[n['g_sens']] = DEFVAL
            data[n['P']] = DEFVAL
            data[n['Q']] = DEFVAL
            data[n['R']] = DEFVAL

            data[n['s2n_w']] = DEFVAL
            data[n['loglike']] = BIG_DEFVAL
            data[n['chi2per']] = PDEFVAL
            data[n['aic']] = BIG_PDEFVAL
            data[n['bic']] = BIG_PDEFVAL
        
        self.data=data



