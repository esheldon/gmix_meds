from .lmfit import *
from .lmfit import _stat_names

from gmix_image.gmix_fit import GMixFitMultiSimpleMB

class MedsFitMB(MedsFit):
    def __init__(self, meds_files, **keys):
        """
        Multi-band fitting

        parameters
        ----------
        meds_file:
            string of meds path
        obj_range: optional
            a 2-element sequence or None.  If not None, only the objects in the
            specified range are processed and get_data() will only return those
            objects. The range is inclusive unlike slices.
        psf_model: string, int
            e.g. "lm2" or in the future "em2"
        det_cat: optional
            Catalog to use as "detection" catalog; an overall flux will be fit
            with best simple model fit from this.
        """

        self.conf={}
        self.conf.update(keys)

        self.meds_files=meds_files
        self.nband=len(meds_files)
        self.iband = range(self.nband)

        self._load_meds_files()
        self.mb_psfex_list = self._get_all_mb_psfex_objects()

        self.obj_range=keys.get('obj_range',None)
        self._set_index_list()

        # in arcsec (or units of jacobian)
        self.use_cenprior=keys.get("use_cenprior",True)
        self.cen_width=keys.get('cen_width',1.0)

        self.gprior=keys.get('gprior',None)

        self.psf_model=keys.get('psf_model','em2')
        self.psf_offset_max=keys.get('psf_offset_max',PSF_OFFSET_MAX)
        self.psf_ngauss=get_psf_ngauss(self.psf_model)

        self.debug=keys.get('debug',0)

        self.psf_ntry=keys.get('psf_ntry', LM_MAX_TRY)
        self.obj_ntry=keys.get('obj_ntry',2)

        self.region=keys.get('region','seg_and_sky')
        self.max_box_size=keys.get('max_box_size',2048)

        self.simple_models=keys.get('simple_models',['exp','dev'])

        self.reject_outliers=keys.get('reject_outliers',False)

        self.make_plots=keys.get('make_plots',False)
        self.prompt=keys.get('prompt',True)

        self._make_struct()

    def get_meds_meta_list(self):
        return [m.copy() for m in self.meds_meta_list]


    def _fit_obj(self, index):
        """
        Process the indicated object through the requested fits
        """

        t0=time.time()

        self.data['flags'][index] = self._mb_obj_check(index)
        if self.data['flags'][index] != 0:
            return 0

        # lists of lists
        mb_imlist,mb_wtlist,self.mb_coadd_list = self._get_mb_imlist_wtlist(index)
        mb_jacob_list=self._get_mb_jacobian_list(index)

        print >>stderr,mb_imlist[0][0].shape
    
        mb_keep_list,mb_psf_gmix_list,flags=self._fit_mb_psfs(index,mb_jacob_list)
        if any(flags):
            self.data['flags'][index,:] = flags
            return
        mb_keep_list,mb_psf_gmix_list,flags=self._remove_mb_bad_psfs(mb_keep_list,mb_psf_gmix_list)
        if any(flags):
            self.data['flags'][index,:] = flags
            return

        mb_imlist, mb_wtlist, mb_jacob_list, len_list = \
            self._extract_sub_lists(mb_keep_list,mb_imlist,mb_wtlist,mb_jacob_list)

        self.data['nimage_use'][index, :] = len_list

        sdata={'mb_imlist':mb_imlist,
               'mb_wtlist':mb_wtlist,
               'mb_jacob_list':mb_jacob_list,
               'mb_psf_gmix_list':mb_psf_gmix_list}

        # cmodel not implemented, nor BD
        if 'psf' in self.conf['fit_types']:
            self._fit_mb_psf_flux(index, sdata)
        if 'simple' in self.conf['fit_types']:
            self._fit_mb_simple_models(index, sdata)

        self.data['time'][index] = time.time()-t0

    def _mb_obj_check(self, index):
        for band in self.iband:
            meds=self.meds_list[band]
            flags=self._obj_check(meds, index)
            if flags != 0:
                break
        return flags

    def _fit_mb_psf_flux(self, index, sdata):
        """
        Perform PSF flux fits on each band separately
        """

        for band in self.iband:
            self._fit_single_psf_flux(index, sdata, band)

    def _fit_single_psf_flux(self, index, sdata, band):
        """
        Fit a single band to the psf model
        """
        if self.debug:
            print >>stderr,'\tfitting psf flux'

        cen_prior=None
        if self.use_cenprior:
            cen_prior=CenPrior([0.0]*2, [self.cen_width]*2)

        gm=GMixFitMultiPSFFlux(sdata['mb_imlist'][band],
                               sdata['mb_wtlist'][band],
                               sdata['mb_jacob_list'][band],
                               sdata['mb_psf_gmix_list'][band],
                               cen_prior=cen_prior,
                               lm_max_try=self.obj_ntry)
        res=gm.get_result()
        self.data['psf_flags'][index,band] = res['flags']
        self.data['psf_iter'][index,band] = res['numiter']
        self.data['psf_tries'][index,band] = res['ntry']

        if res['flags']==0:
            self.data['psf_pars'][index,band,:]=res['pars']
            self.data['psf_pars_cov'][index,band,:,:] = res['pcov']

            flux=res['pars'][2]
            flux_err=sqrt(res['pcov'][2,2])
            self.data['psf_flux'][index,band] = flux
            self.data['psf_flux_err'][index,band] = flux_err

            print >>stderr,'    psf_flux: %g +/- %g' % (flux,flux_err)

            n=get_model_names('psf')
            for sn in _stat_names:
                self.data[n[sn]][index,band] = res[sn]

            if self.debug:
                fmt='\t\t%s: %g +/- %g'
                print >>stderr,fmt % ('psf_flux',flux,flux_err)


    def _fit_mb_simple_models(self, index, sdata):
        """
        Fit all the simple models
        """

        for model in self.simple_models:
            print >>stderr,'    fitting:',model,

            gm=self._fit_mb_simple(index, model, sdata)

            res=gm.get_result()

            print >>stderr,'ntries:',res['ntry']

            self._copy_mb_simple_pars(index, res)
            self._print_fluxes(res)
            if self.debug:
                self._print_pcov(res)


    def _fit_mb_simple(self, index, model, sdata):
        """
        Fit one of the "simple" models, e.g. exp or dev
        """


        if sdata['mb_imlist'][0][0].shape[0] >= 128:
            ntry=2
        else:
            ntry=self.obj_ntry

        for i in xrange(ntry):
            guess=self._get_simple_guess(index, sdata)
            cen_prior=None
            if self.use_cenprior:
                cen_prior=CenPrior(guess[0:0+2], [self.cen_width]*2)

            gm=GMixFitMultiSimpleMB(sdata['mb_imlist'],
                                    sdata['mb_wtlist'],
                                    sdata['mb_jacob_list'],
                                    sdata['mb_psf_gmix_list'],
                                    guess,
                                    model,
                                    cen_prior=cen_prior,
                                    gprior=self.gprior)
            res=gm.get_result()
            if res['flags']==0:
                break

        res['ntry'] = i+1
        return gm


    def _get_simple_guess(self, index, sdata):
        npars=5+self.nband
        guess=numpy.zeros(npars)
        guess[0:0+2] = 0.0

        gtot = 0.8*numpy.random.random()
        theta=numpy.random.random()*numpy.pi
        g1rand = gtot*numpy.cos(2*theta)
        g2rand = gtot*numpy.sin(2*theta)
        guess[2]=g1rand
        guess[3]=g2rand

        # this is a terrible guess
        guess[4] = 16.0*(1.0 + srandu())

        for band in self.iband:
            if self.data['psf_flags'][index,band]==0:
                counts_guess=2*self.data['psf_flux'][index,band]
            else:
                # terrible guess
                im_list=sdata['mb_imlist'][band]
                cvals=[im.sum() for im in im_list]
                counts_guess=numpy.median(cvals)
            guess[5+band] = counts_guess*(1.0 + srandu())

        return guess


    def _copy_mb_simple_pars(self, index, res):
        model=res['model']
        n=get_model_names(model)

        self.data[n['flags']][index] = res['flags']
        self.data[n['iter']][index] = res['numiter']
        self.data[n['tries']][index] = res['ntry']

        if res['flags'] == 0:
            self.data[n['pars']][index,:] = res['pars']
            self.data[n['pars_cov']][index,:,:] = res['pcov']

            self.data[n['flux']][index] = res['Flux']
            self.data[n['flux_err']][index] = res['Flux_err']
            self.data[n['flux_cov']][index] = res['Flux_cov']

            self.data[n['g']][index,:] = res['g']
            self.data[n['g_cov']][index,:,:] = res['g_cov']

            for sn in _stat_names:
                self.data[n[sn]][index] = res[sn]

    def _print_fluxes(self, res):
        from gmix_image.util import print_pars
        if res['flags']==0:
            print_pars(res['Flux'], stream=stderr, front='        ')
            print_pars(res['Flux_err'], stream=stderr, front='        ')

    def _print_pcov(self, res):
        if res['flags']==0:
            import images
            import esutil as eu
            images.imprint(eu.stat.cov2cor( res['pcov']) , fmt='%10f')

    def _extract_sub_lists(self,
                           mb_keep_list0,
                           mb_imlist0,
                           mb_wtlist0,
                           mb_jacob_list0):

        mb_imlist=[]
        mb_wtlist=[]
        mb_jacob_list=[]
        len_list=[]
        for band in self.iband:
            keep_list = mb_keep_list0[band]

            imlist0 = mb_imlist0[band]
            wtlist0 = mb_wtlist0[band]
            jacob_list0 = mb_jacob_list0[band]

            imlist = [imlist0[i] for i in keep_list]
            wtlist = [wtlist0[i] for i in keep_list]
            jacob_list = [jacob_list0[i] for i in keep_list]

            mb_imlist.append( imlist )
            mb_wtlist.append( wtlist )
            mb_jacob_list.append( jacob_list )

            len_list.append( len(imlist) )

        return mb_imlist, mb_wtlist, mb_jacob_list, len_list

    def _fit_mb_psfs(self, index, mb_jacob_list):
        mb_keep_list=[]
        mb_gmix_list=[]

        flags=[]
        for band in self.iband:
            meds=self.meds_list[band]
            jacob_list=mb_jacob_list[band]
            psfex_list=self.mb_psfex_list[band]

            keep_list, gmix_list = self._fit_psfs(meds,index,jacob_list,psfex_list)

            mb_keep_list.append( keep_list )
            mb_gmix_list.append( gmix_list )

            if len(keep_list) == 0:
                flags.append( PSF_FIT_FAILURE )
            else:
                flags.append( 0 )

       
        return mb_keep_list, mb_gmix_list, flags

    def _remove_mb_bad_psfs(self, mb_keep_list, mb_gmix_list):
        flags=[]
        for band in self.iband:
            keep_list0=mb_keep_list[band]
            gmix_list0=mb_gmix_list[band]

            keep_list,gmix_list=self._remove_bad_psfs(keep_list0,gmix_list0)

            mb_keep_list[band] = keep_list
            mb_gmix_list[band] = gmix_list

            if len(keep_list) == 0:
                flags.append( PSF_LARGE_OFFSETS )
            else:
                flags.append( 0 )

        return mb_keep_list, mb_gmix_list, flags
        
    def _get_mb_imlist_wtlist(self, index):
        """
        Get a list of the jocobians for this object
        skipping the coadd
        """

        mb_imlist=[]
        mb_wtlist=[]
        mb_coadd_list=[]

        for band in self.iband:
            meds=self.meds_list[band]

            # inherited functions
            imlist,coadd=self._get_imlist(meds,index)
            wtlist=self._get_wtlist(meds,index)

            self.data['nimage_tot'][index,band] = len(imlist)

            mb_imlist.append(imlist)
            mb_wtlist.append(wtlist)
            mb_coadd_list.append(coadd)

        
        return mb_imlist,mb_wtlist, mb_coadd_list

    def _get_mb_jacobian_list(self, index):
        """
        Get a list of the jocobians for this object
        skipping the coadd
        """

        mb_jacob_list=[]
        for band in self.iband:
            meds=self.meds_list[band]

            jacob_list = self._get_jacobian_list(meds,index)
            mb_jacob_list.append(jacob_list)

        return mb_jacob_list
 

    def _get_all_mb_psfex_objects(self):
        """
        Load psfex objects for each of the SE images
        include the coadd so we get  the index right
        """
        desdata=os.environ['DESDATA']
        meds_desdata=self.meds_list[0]._meta['DESDATA'][0]

        mb_psfex_list=[]

        for band in self.iband:
            meds=self.meds_list[band]

            psfex_list = self._get_all_psfex_objects(meds)
            mb_psfex_list.append( psfex_list )

        return mb_psfex_list

    def _load_meds_files(self):
        """
        Load all listed meds files
        """

        self.meds_list=[]
        self.meds_meta_list=[]

        for i,f in enumerate(self.meds_files):
            print >>stderr,f
            medsi=meds.MEDS(f)
            medsi_meta=medsi.get_meta()
            if i==0:
                self.nobj=medsi.size
            else:
                nobj=medsi.size
                if nobj != self.nobj:
                    raise ValueError("mismatch in meds "
                                     "sizes: %d/%d" % (self.nobj,nobj))
            self.meds_list.append(medsi)
            self.meds_meta_list.append(medsi_meta)

    def _make_struct(self):
        nband=self.nband

        dt=[('id','i4'),
            ('flags','i4'),
            ('nimage_tot','i4',nband),
            ('nimage_use','i4',nband),
            ('time','f8')]


        simple_npars=5+nband
        simple_models=self.simple_models

        psf_npars_perband=3

        for model in simple_models:
            n=get_model_names(model)

            dt+=[(n['flags'],'i4'),
                 (n['iter'],'i4'),
                 (n['tries'],'i4'),
                 (n['pars'],'f8',simple_npars),
                 (n['pars_cov'],'f8',(simple_npars,simple_npars)),
                 (n['flux'],'f8',nband),
                 (n['flux_err'],'f8',nband),
                 (n['flux_cov'],'f8',(nband,nband)),
                 (n['g'],'f8',2),
                 (n['g_cov'],'f8',(2,2)),
                
                 (n['s2n_w'],'f8'),
                 (n['loglike'],'f8'),
                 (n['chi2per'],'f8'),
                 (n['dof'],'f8'),
                 (n['fit_prob'],'f8'),
                 (n['aic'],'f8'),
                 (n['bic'],'f8'),
                ]

        # the psf fits are done for each band separately
        n=get_model_names('psf')
        dt += [('psf_flags','i4',nband),
               ('psf_iter','i4',nband),
               ('psf_tries','i4',nband),
               ('psf_pars','f8',(nband,psf_npars_perband)),
               ('psf_pars_cov','f8',(nband,psf_npars_perband,psf_npars_perband)),
               ('psf_flux','f8',nband),
               ('psf_flux_err','f8',nband),
               (n['s2n_w'],'f8',nband),
               (n['loglike'],'f8',nband),
               (n['chi2per'],'f8',nband),
               (n['dof'],'f8',nband),
               (n['fit_prob'],'f8',nband),
               (n['aic'],'f8',nband),
               (n['bic'],'f8',nband)]

        data=numpy.zeros(self.nobj, dtype=dt)
        data['id'] = 1+numpy.arange(self.nobj)


        for model in simple_models:
            n=get_model_names(model)

            data[n['flags']] = NO_ATTEMPT

            data[n['pars']] = DEFVAL
            data[n['pars_cov']] = PDEFVAL
            data[n['flux']] = DEFVAL
            data[n['flux_err']] = PDEFVAL
            data[n['flux_cov']] = PDEFVAL
            data[n['g']] = DEFVAL
            data[n['g_cov']] = PDEFVAL

            data[n['s2n_w']] = DEFVAL
            data[n['loglike']] = BIG_DEFVAL
            data[n['chi2per']] = PDEFVAL
            data[n['aic']] = BIG_PDEFVAL
            data[n['bic']] = BIG_PDEFVAL

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
       
        self.data=data

