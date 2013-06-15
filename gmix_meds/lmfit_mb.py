from .lmfit import *
from .lmfit import _stat_names

from gmix_image.gmix_mcmc import MixMCSimpleMB


class MedsFitMB(object):
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

    def _fit_obj(self, index):
        """
        Process the indicated object through the requested fits
        """

        t0=time.time()

        self.data['flags'][index] = self._obj_check(index)
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

    def _extract_sub_lists(self,
                           mb_keep_list0,
                           mb_imlist0,
                           mb_wtlist0,
                           mb_jacob_list0):

        mb_imlist=[]
        mb_wtlist=[]
        mb_jacob_list=[]
        len_list=[]
        for band in xrange(self.band):
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

        return mb_imlist, mb_wtlist, mb_jacob_list

    def _fit_mb_psfs(self, index, mb_jacob_list):
        mb_keep_list=[]
        mb_gmix_list=[]

        flags=0
        for band in self.iband:
            meds=self.meds_list[band]
            jacob_list=mb_jacob_list[band]

            keep_list, gmix_list = self._fit_psfs(meds,index,jacob_list)

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
        meds_desdata=self.meds._meta['DESDATA'][0]

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
        self.meta_meta_list=[]

        for i,f in enumerate(self.meds_files):
            print >>stderr,f
            meds=meds.MEDS(f)
            meds_meta=meds.get_meta()
            if i==0:
                self.nobj=meds.size
            else:
                nobj=meds.size
                if nobj != self.nobj:
                    raise ValueError("mismatch in meds "
                                     "sizes: %d/%d" % (self.nobj,nobj))
            self.meds_list.append(meds)
            self.meds_meta_list.append(meds_meta)

    def _make_struct(self):
        nband=self.nband

        dt=[('id','i4'),
            ('flags','i4'),
            ('nimage_tot','i4',nband),
            ('nimage_use','i4',nband),
            ('time','f8')]


        simple_npars=5+nband
        simple_models=self.simple_models

        psf_npars=2+nband

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
               ('psf_pars','f8',nband,psf_npars),
               ('psf_pars_cov','f8',(nband,psf_npars,psf_npars)),
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

