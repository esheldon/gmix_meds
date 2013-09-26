from .lmfit import *
from .lmfit import _stat_names

from gmix_image.gmix_fit import GMixFitMultiSimpleMB
from gmix_image.gmix_fit import GMixFitMultiBD

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
        self.checkpoint = self.conf.get('checkpoint',172800)
        self.checkpoint_file = self.conf.get('checkpoint_file',None)

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

        self.max_simple_time=self.conf.get('max_simple_time', 125.0)

        self.debug=keys.get('debug',0)

        self.psf_ntry=keys.get('psf_ntry', LM_MAX_TRY)
        self.obj_ntry=keys.get('obj_ntry',2)

        self.region=keys.get('region','seg_and_sky')
        self.max_box_size=keys.get('max_box_size',2048)

        self.simple_models=keys.get('simple_models',SIMPLE_MODELS_DEFAULT )

        self.reject_outliers=keys.get('reject_outliers',False)

        self.make_plots=keys.get('make_plots',False)
        self.prompt=keys.get('prompt',True)

        self._checkpoint_data=keys.get('checkpoint_data',None)

        if self._checkpoint_data is not None:
            self.data=self._checkpoint_data
        else:
            self._make_struct()


    def get_meds_meta_list(self):
        return [m.copy() for m in self.meds_meta_list]


    def _fit_obj(self, dindex):
        """
        Process the indicated object through the requested fits
        """

        t0=time.time()

        # for checkpointing
        self.data['processed'][dindex]=1

        mindex = self.index_list[dindex]

        self.data['id'][dindex] = self.meds_list[0]['number'][mindex]

        self.data['flags'][dindex] = self._mb_obj_check(mindex)
        if self.data['flags'][dindex] != 0:
            return 0

        # lists of lists
        mb_imlist,mb_wtlist,self.mb_coadd_list = self._get_mb_imlist_wtlist(dindex,mindex)
        mb_jacob_list=self._get_mb_jacobian_list(mindex)

        print >>stderr,mb_imlist[0][0].shape
    
        mb_keep_list,mb_psf_gmix_list,flags=self._fit_mb_psfs(mindex,mb_jacob_list)
        if any(flags):
            self.data['flags'][dindex] = PSF_FIT_FAILURE 
            return
        mb_keep_list,mb_psf_gmix_list,flags=self._remove_mb_bad_psfs(mb_keep_list,mb_psf_gmix_list)
        if any(flags):
            self.data['flags'][dindex] = PSF_LARGE_OFFSETS 
            return

        mb_imlist, mb_wtlist, mb_jacob_list, len_list = \
            self._extract_sub_lists(mb_keep_list,mb_imlist,mb_wtlist,mb_jacob_list)
        
        self._set_median_im_sums(mb_imlist)

        self.data['nimage_use'][dindex, :] = len_list

        sdata={'mb_keep_list':mb_keep_list,
               'mb_imlist':mb_imlist,
               'mb_wtlist':mb_wtlist,
               'mb_jacob_list':mb_jacob_list,
               'mb_psf_gmix_list':mb_psf_gmix_list}

        self._do_all_fits(dindex, sdata)

        self.data['time'][dindex] = time.time()-t0

    def _do_all_fits(self, dindex, sdata):
        if 'psf' in self.conf['fit_types']:
            self._fit_mb_psf_flux(dindex, sdata)
        else:
            raise ValueError("you should do a psf_flux fit")
        
        if 'psf1' in self.conf['fit_types']:
            self._fit_mb_psf1_flux(dindex, sdata)
 
        max_psf_s2n=self.data['psf_flux_s2n'][dindex,:].max()
        if max_psf_s2n >= self.conf['min_psf_s2n']:
            if 'simple' in self.conf['fit_types']:
                self._fit_mb_simple_models(dindex, sdata)
            if 'bd' in self.conf['fit_types']:
                self._fit_bd(dindex, sdata)
        else:
            mess="    psf s/n too low: %s (%s)"
            mess=mess % (max_psf_s2n,self.conf['min_psf_s2n'])
            print >>stderr,mess

    def _mb_obj_check(self, mindex):
        for band in self.iband:
            meds=self.meds_list[band]
            flags=self._obj_check(meds, mindex)
            if flags != 0:
                break
        return flags

    def _fit_mb_psf_flux(self, dindex, sdata):
        """
        Perform PSF flux fits on each band separately
        """

        print >>stderr,'    fitting: psf'
        for band in self.iband:
            self._fit_single_psf_flux(dindex, sdata, band)

        print_pars(self.data['psf_flux'][dindex], stream=stderr, front='        ')

        if any( map(lambda x: not x, self.data['psf_flags'][dindex]) ):
            print_pars(self.data['psf_flux_err'][dindex], stream=stderr, front='        ')

    def _fit_mb_psf1_flux(self, dindex, sdata):
        """
        Perform PSF flux fits on a single SE image in each band separately
        """

        im_index = self.conf['psf1_index']
        print >>stderr,'    fitting: psf1 at index',im_index
        for band in self.iband:
            self._fit_psf1_flux(dindex, sdata, band, im_index)

        print_pars(self.data['psf1_flux'][dindex], stream=stderr, front='        ')

        if any( map(lambda x: not x, self.data['psf1_flags'][dindex]) ):
            print_pars(self.data['psf1_flux_err'][dindex], stream=stderr, front='        ')



    def _fit_single_psf_flux(self, dindex, sdata, band):
        """
        Fit a single band to the psf model
        """

        cen_prior=None
        if self.use_cenprior:
            cen_prior=CenPrior([0.0]*2, [self.cen_width]*2)

        counts_guess=self.median_im_sums[band]
        if counts_guess < 0:
            counts_guess=1.0

        gm=GMixFitMultiPSFFlux(sdata['mb_imlist'][band],
                               sdata['mb_wtlist'][band],
                               sdata['mb_jacob_list'][band],
                               sdata['mb_psf_gmix_list'][band],
                               cen_prior=cen_prior,
                               lm_max_try=self.obj_ntry,
                               counts_guess=counts_guess)
        res=gm.get_result()
        self.data['psf_flags'][dindex,band] = res['flags']
        self.data['psf_iter'][dindex,band] = res['numiter']
        self.data['psf_tries'][dindex,band] = res['ntry']

        if res['flags']==0:
            self.data['psf_pars'][dindex,band,:]=res['pars']
            self.data['psf_pars_cov'][dindex,band,:,:] = res['pcov']

            flux=res['pars'][2]
            flux_err=sqrt(res['pcov'][2,2])
            self.data['psf_flux'][dindex,band] = flux
            self.data['psf_flux_err'][dindex,band] = flux_err

            self.data['psf_flux_s2n'][dindex,band] = flux/flux_err


            n=get_model_names('psf')
            for sn in _stat_names:
                self.data[n[sn]][dindex,band] = res[sn]

    def _fit_psf1_flux(self, dindex, sdata, band, im_index):
        """
        Fit a single band to the psf model
        """

        cen_prior=None
        if self.use_cenprior:
            cen_prior=CenPrior([0.0]*2, [self.cen_width]*2)

        # im_index does not include coadd currently
        im_index0 = im_index-1
        if im_index0 not in sdata['mb_keep_list'][band]:
            print >>stderr,'    image at index',im_index,'not used for band',band
            self.data['psf1_flags'][dindex,band] = PSF1_NOT_KEPT
            return

        imlist = [sdata['mb_imlist'][band][im_index0]]
        wtlist = [sdata['mb_wtlist'][band][im_index0]]
        jlist = [sdata['mb_jacob_list'][band][im_index0]]
        psflist = [sdata['mb_psf_gmix_list'][band][im_index0]]

        counts_guess = imlist[0].sum()
        if counts_guess < 0:
            counts_guess=1.0

        gm=GMixFitMultiPSFFlux(imlist,
                               wtlist,
                               jlist,
                               psflist,
                               cen_prior=cen_prior,
                               lm_max_try=self.obj_ntry,
                               counts_guess=counts_guess)
        res=gm.get_result()
        self.data['psf1_flags'][dindex,band] = res['flags']
        self.data['psf1_iter'][dindex,band] = res['numiter']
        self.data['psf1_tries'][dindex,band] = res['ntry']

        if res['flags']==0:
            self.data['psf1_pars'][dindex,band,:]=res['pars']
            self.data['psf1_pars_cov'][dindex,band,:,:] = res['pcov']

            flux=res['pars'][2]
            flux_err=sqrt(res['pcov'][2,2])
            self.data['psf1_flux'][dindex,band] = flux
            self.data['psf1_flux_err'][dindex,band] = flux_err

            self.data['psf1_flux_s2n'][dindex,band] = flux/flux_err

            n=get_model_names('psf1')
            for sn in _stat_names:
                self.data[n[sn]][dindex,band] = res[sn]


    def _fit_mb_simple_models(self, dindex, sdata):
        """
        Fit all the simple models
        """

        for model in self.simple_models:
            print >>stderr,'    fitting:',model,

            gm=self._fit_mb_simple(dindex, model, sdata)

            res=gm.get_result()

            print >>stderr,'ntries:',res['ntry']

            self._copy_mb_pars(dindex, res)
            self._print_fluxes(res)
            if self.debug > 1:
                self._print_pcov(res)


    def _fit_mb_simple(self, dindex, model, sdata):
        """
        Fit one of the "simple" models, e.g. exp or dev
        """

        tm0=time.time()

        box_size=sdata['mb_imlist'][0][0].shape[0]
        if box_size >= 128:
            ntry=2
        elif box_size >= 96:
            ntry=4
        else:
            ntry=self.obj_ntry

        for i in xrange(ntry):
            guess=self._get_simple_guess(dindex, sdata)
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

            t=time.time()-tm0
            if t > self.max_simple_time:
                res['flags'] |= ALGO_TIMEOUT
                break

        res['ntry'] = i+1
        return gm


    def _get_simple_guess(self, dindex, sdata):
        """
        Guesses for simple models

        The size guess is pretty stupid
        """

        npars=5+self.nband
        guess=numpy.zeros(npars)
        guess[0:0+2] = 0.1*srandu(2)

        gtot = 0.8*numpy.random.random()
        theta=numpy.random.random()*numpy.pi
        g1rand = gtot*numpy.cos(2*theta)
        g2rand = gtot*numpy.sin(2*theta)
        guess[2]=g1rand
        guess[3]=g2rand

        # this is a terrible guess
        guess[4] = 16.0*(1.0 + 0.2*srandu())

        for band in self.iband:
            if self.data['psf_flags'][dindex,band]==0:
                counts_guess=2*self.data['psf_flux'][dindex,band]
            else:
                # terrible guess
                counts_guess=self.median_im_sums[band]
                if counts_guess < 0:
                    counts_guess=1.0

            guess[5+band] = counts_guess*(1.0 + 0.2*srandu())

        return guess


    def _copy_mb_pars(self, dindex, res):
        model=res['model']
        n=get_model_names(model)

        self.data[n['flags']][dindex] = res['flags']
        self.data[n['iter']][dindex] = res['numiter']
        self.data[n['tries']][dindex] = res['ntry']

        if res['flags'] == 0:
            self.data[n['pars']][dindex,:] = res['pars']
            self.data[n['pars_cov']][dindex,:,:] = res['pcov']

            self.data[n['flux']][dindex] = res['flux']
            self.data[n['flux_cov']][dindex] = res['flux_cov']

            self.data[n['g']][dindex,:] = res['g']
            self.data[n['g_cov']][dindex,:,:] = res['g_cov']

            for sn in _stat_names:
                self.data[n[sn]][dindex] = res[sn]

    def _fit_bd(self, dindex, sdata):
        """
        Fit all the simple models
        """

        print >>stderr,'    fitting: bd'

        box_size=sdata['mb_imlist'][0][0].shape[0]
        if box_size >= 128:
            ntry=2
        elif box_size >= 96:
            ntry=4
        else:
            ntry=self.obj_ntry


        for i in xrange(ntry):
            guess=self._get_bd_guess(dindex, sdata)
            if self.debug:
                print_pars(guess,front='    bd guess:',stream=stderr)

            cen_prior=None
            if self.use_cenprior:
                cen_prior=CenPrior(guess[0:0+2], [self.cen_width]*2)

            gm=GMixFitMultiBD(sdata['mb_imlist'],
                              sdata['mb_wtlist'],
                              sdata['mb_jacob_list'],
                              sdata['mb_psf_gmix_list'],
                              guess,
                              cen_prior=cen_prior,
                              gprior=self.gprior)
            res=gm.get_result()
            if res['flags']==0:
                break

        res['ntry'] = i+1

        print >>stderr,'ntries:',res['ntry']

        self._copy_mb_pars(dindex, res)

        self._print_fluxes(res)
        if self.debug > 1:
            self._print_pcov(res)

    def _get_bd_guess(self, dindex, sdata):
        """
        Guesses for bulge+disk

        The size guess is pretty stupid
        """
        npars=6+2*self.nband

        guess=numpy.zeros(npars)
        guess[0:0+2] = 0.1*srandu(2)

        gtot = 0.8*numpy.random.random()
        theta=numpy.random.random()*numpy.pi
        g1rand = gtot*numpy.cos(2*theta)
        g2rand = gtot*numpy.sin(2*theta)
        guess[2]=g1rand
        guess[3]=g2rand

        frac=0.2

        guess[4:4+2] = self._get_bd_T_guess(dindex)
        guess[4:4+2] = guess[4:4+2]*(1.0+frac*srandu(2))

        for band in self.iband:

            fluxes=self._get_bd_flux_guess(dindex, band)
            guess[6+2*band:6+2*band+2] = fluxes

            guess[6+2*band:6+2*band+2] *= (1.0 + frac*srandu(2))

        return guess

    def _get_bd_T_guess(self, dindex):

        trymodels=['exp','dev']
        Tvals=[]
        for model in trymodels:
            if model in self.simple_models:
                flagn='%s_flags' % model
                if self.data[flagn][dindex]==0:
                    pn='%s_pars' % model
                    T = self.data[pn][dindex,4]
                    Tvals.append(T)
 
        ng=len(Tvals)                    
        if ng==0:
            # terrible!
            Tvals=[16]

        if len(Tvals)==1:
            portion_b = 0.5 + 0.3*srandu()
            portion_d = 1.0 - portion_b

            Tvals = [ portion_b*Tvals[0], portion_d*Tvals[0] ]

        return Tvals

    def _get_bd_flux_guess(self, dindex, band):

        dosplit_flux=True
        dopsf=False

        trymodels=['exp','dev']
        fluxes=[]
        for model in trymodels:
            if model in self.simple_models:
                flagn='%s_flags' % model
                if self.data[flagn][dindex]==0:
                    fluxn='%s_flux' % model
                    c = self.data[fluxn][dindex,band]
                    fluxes.append(c)
                
        ng=len(fluxes)                    
        if ng==0:
            if self.data['psf_flags'][dindex,band]==0:
                fluxes=[2*self.data['psf_flux'][dindex,band]]
            else:
                # terrible guess
                fluxes=[self.median_im_sums[band]]

        if len(fluxes)==1:
            portion_b = 0.5 + 0.3*srandu()
            portion_d = 1.0 - portion_b

            fluxes = [ portion_b*fluxes[0], portion_d*fluxes[0] ]

        return fluxes


    def _print_fluxes(self, res):
        if res['flags']==0:
            print_pars(res['flux'], stream=stderr, front='        ')
            flux_err=sqrt(diag(res['flux_cov']))
            print_pars(flux_err, stream=stderr, front='        ')

    def _print_pcov(self, res):
        if res['flags']==0:
            import images
            import esutil as eu
            images.imprint(eu.stat.cov2cor( res['pcov']) , fmt='%10f')

    def _set_median_im_sums(self,mb_imlist):
        """
        One for each band
        """
        self.median_im_sums=[]
        for band in self.iband:
            im_list=mb_imlist[band]
            cvals=[im.sum() for im in im_list]
            med=numpy.median(cvals)
            self.median_im_sums.append(med)


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

    def _fit_mb_psfs(self, mindex, mb_jacob_list):
        mb_keep_list=[]
        mb_gmix_list=[]

        flags=[]
        for band in self.iband:
            meds=self.meds_list[band]
            jacob_list=mb_jacob_list[band]
            psfex_list=self.mb_psfex_list[band]

            keep_list, gmix_list = self._fit_psfs(meds,mindex,jacob_list,psfex_list)

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
        
    def _get_mb_imlist_wtlist(self, dindex, mindex):
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
            imlist,coadd=self._get_imlist(meds,mindex)
            wtlist=self._get_wtlist(meds,mindex)

            self.data['nimage_tot'][dindex,band] = len(imlist)

            mb_imlist.append(imlist)
            mb_wtlist.append(wtlist)
            mb_coadd_list.append(coadd)

        
        return mb_imlist,mb_wtlist, mb_coadd_list

    def _get_mb_jacobian_list(self, mindex):
        """
        Get a list of the jocobians for this object
        skipping the coadd
        """

        mb_jacob_list=[]
        for band in self.iband:
            meds=self.meds_list[band]

            jacob_list = self._get_jacobian_list(meds,mindex)
            mb_jacob_list.append(jacob_list)

        return mb_jacob_list
 

    def _get_all_mb_psfex_objects(self):
        """
        Load psfex objects for each of the SE images
        include the coadd so we get  the index right
        """
        print 'loading psfex'
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
                nobj_tot=medsi.size
            else:
                nobj=medsi.size
                if nobj != nobj_tot:
                    raise ValueError("mismatch in meds "
                                     "sizes: %d/%d" % (nobj_tot,nobj))
            self.meds_list.append(medsi)
            self.meds_meta_list.append(medsi_meta)

        self.nobj = self.meds_list[0].size

    def _make_struct(self):
        nband=self.nband

        dt=[('id','i4'),
            ('processed','i1'),
            ('flags','i4'),
            ('nimage_tot','i4',nband),
            ('nimage_use','i4',nband),
            ('time','f8')]

        
        if 'bd' in self.conf['fit_types']:
            do_bd=True
        else:
            do_bd=False

        simple_npars=5+nband
        simple_models=self.simple_models

        psf_npars_perband=3

        bd_npars = 6 + 2*self.nband

        all_models = [s for s in simple_models]
        if do_bd:
            all_models.append('bd')

        for model in all_models:
            n=get_model_names(model)

            if model=='bd':
                np=bd_npars
                npband=2*nband
            else:
                np=simple_npars
                npband=nband

            dt+=[(n['flags'],'i4'),
                 (n['iter'],'i4'),
                 (n['tries'],'i4'),
                 (n['pars'],'f8',np),
                 (n['pars_cov'],'f8',(np,np)),
                 (n['flux'],'f8',npband),
                 (n['flux_cov'],'f8',(npband,npband)),
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
               ('psf_flux_s2n','f8',nband),
               (n['s2n_w'],'f8',nband),
               (n['loglike'],'f8',nband),
               (n['chi2per'],'f8',nband),
               (n['dof'],'f8',nband),
               (n['fit_prob'],'f8',nband),
               (n['aic'],'f8',nband),
               (n['bic'],'f8',nband)]

        # fit to one of the SE images or coadd
        if 'psf1' in self.conf['fit_types']:
            n=get_model_names('psf1')
            dt += [('psf1_flags','i4',nband),
                   ('psf1_iter','i4',nband),
                   ('psf1_tries','i4',nband),
                   ('psf1_pars','f8',(nband,psf_npars_perband)),
                   ('psf1_pars_cov','f8',(nband,psf_npars_perband,psf_npars_perband)),
                   ('psf1_flux','f8',nband),
                   ('psf1_flux_err','f8',nband),
                   ('psf1_flux_s2n','f8',nband),
                   (n['s2n_w'],'f8',nband),
                   (n['loglike'],'f8',nband),
                   (n['chi2per'],'f8',nband),
                   (n['dof'],'f8',nband),
                   (n['fit_prob'],'f8',nband),
                   (n['aic'],'f8',nband),
                   (n['bic'],'f8',nband)]


        num=self.index_list.size
        data=numpy.zeros(num, dtype=dt)
        #data['id'] = 1+self.index_list


        for model in all_models:
            n=get_model_names(model)

            data[n['flags']] = NO_ATTEMPT

            data[n['pars']] = DEFVAL
            data[n['pars_cov']] = PDEFVAL
            data[n['flux']] = DEFVAL
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
        data['psf_flux_s2n'] = DEFVAL

        data['psf_s2n_w'] = DEFVAL
        data['psf_loglike'] = BIG_DEFVAL
        data['psf_chi2per'] = PDEFVAL
        data['psf_aic'] = BIG_PDEFVAL
        data['psf_bic'] = BIG_PDEFVAL
       
        if 'psf1' in self.conf['fit_types']:
            data['psf1_flags'] = NO_ATTEMPT
            data['psf1_pars'] = DEFVAL
            data['psf1_pars_cov'] = PDEFVAL
            data['psf1_flux'] = DEFVAL
            data['psf1_flux_err'] = PDEFVAL
            data['psf1_flux_s2n'] = DEFVAL

            data['psf1_s2n_w'] = DEFVAL
            data['psf1_loglike'] = BIG_DEFVAL
            data['psf1_chi2per'] = PDEFVAL
            data['psf1_aic'] = BIG_PDEFVAL
            data['psf1_bic'] = BIG_PDEFVAL
     
        self.data=data

