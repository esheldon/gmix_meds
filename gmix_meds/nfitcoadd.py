from __future__ import print_function
from .nfit import *

class MedsFitCoadd(MedsFit):
    def __init__(self,
                 conf,
                 priors,
                 meds_files,
                 obj_range=None,
                 checkpoint_file=None,
                 checkpoint_data=None):
        """
        Model fitting

        parameters
        ----------
        conf: dict
            Configuration data.  See examples in config/.
        priors: dict
            should contain:
                cen_prior, g_priors, T_priors, counts_priors
        meds_files:
            string or list of of MEDS file path(s)
        """

        self.update(conf)
        self._set_some_defaults()
        
        self.meds_files=get_as_list(meds_files)

        self['nband']=len(self.meds_files)
        self.iband = range(self['nband'])

        self._unpack_priors(priors)

        self._load_meds_files()
        self._load_coadd_cat_files()

        self.obj_range=obj_range
        self._set_index_list()

        self.psfex_list = self._get_psfex_list()

        self.checkpoint_file=checkpoint_file
        self.checkpoint_data=checkpoint_data
        self._setup_checkpoints()

        self.random_state=numpy.random.RandomState()

        if self.checkpoint_data is None:
            self._make_struct()

    def get_epoch_data(self):
        """
        currently None
        """
        return None

    def _get_psfex_list(self):
        """
        Load psfex objects
        """
        print('loading psfex')

        psfex_list=[]
        for band in self.iband:
            meds=self.meds_list[band]

            pex = self._get_psfex_object(meds)
            psfex_list.append( pex )

        return psfex_list

    def _get_psfex_object(self, meds):
        """
        Load psfex objects for each of the SE images
        include the coadd so we get  the index right
        """

        info=meds.get_image_info()

        impath=info['image_path'][0].strip()

        psfpath = self._psfex_path_from_image_path(meds, impath)

        pex=psfex.PSFEx(psfpath)

        return pex


    def _psfex_path_from_image_path(self, meds, image_path):
        """
        infer the psfex path from the image path
        """
        desdata=os.environ['DESDATA']
        meds_desdata=meds._meta['DESDATA'][0]

        psfpath=image_path.replace('.fits.fz','_psfcat.psf')

        if desdata not in psfpath:
            psfpath=psfpath.replace(meds_desdata,desdata)

        return psfpath

    def fit_obj(self, dindex):
        """
        Process the indicated object through the requested fits
        """

        t0=time.time()

        self.dindex=dindex

        # for checkpointing
        self.data['processed'][dindex]=1

        mindex = self.index_list[dindex]

        ncutout_tot=self._get_object_ncutout(mindex)

        # need to do this because we work on subset files
        self.data['id'][dindex] = self.meds_list[0]['id'][mindex]
        self.data['number'][dindex] = self.meds_list[0]['number'][mindex]
        self.data['box_size'][dindex] = \
                self.meds_list[0]['box_size'][mindex]

        flags = self._obj_check(mindex)
        if flags != 0:
            self.data['flags'][dindex] = flags
            return 0

        # MultiBandObsList obects
        mb_obs_list = self._get_multi_band_observations(mindex)

        if len(mb_obs_list) == 0:
            print("  not all bands had at least one psf fit succeed")
            self.data['flags'][dindex] = PSF_FIT_FAILURE 
            return

        print(mb_obs_list[0][0].image.shape)

        self.mb_obs_list = mb_obs_list

        try:
            flags=self._fit_all_models()
        except UtterFailure as err:
            print("Got utter failure error: %s" % str(err))
            flags=UTTER_FAILURE

        self.data['flags'][dindex] = flags
        self.data['time'][dindex] = time.time()-t0

    def _obj_check(self, mindex):
        """
        Check box sizes, number of cutouts

        Require good in all bands
        """
        for band in self.iband:
            flags=self._obj_check_one(band, mindex)
            if flags != 0:
                break
        return flags

    def _obj_check_one(self, band, mindex):
        """
        Check box sizes, number of cutouts, flags on images
        """
        flags=0

        meds=self.meds_list[band]

        # need coadd and at lease one SE image
        ncutout=meds['ncutout'][mindex]
        if ncutout < 1:
            print('No cutouts')
            flags |= NO_CUTOUTS
 
        box_size=meds['box_size'][mindex]
        if box_size > self['max_box_size']:
            print('Box size too big:',box_size)
            flags |= BOX_SIZE_TOO_BIG

        return flags

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

        n_se_images=len(self.mb_obs_list)
         
        if max_s2n >= self['min_psf_s2n'] and len(self['fit_models']) > 0:
            for model in self['fit_models']:
                print('    fitting:',model)
                self._run_model_fit(model)
        else:
            mess="    psf s/n too low: %s (%s)"
            mess=mess % (max_s2n,self['min_psf_s2n'])
            print(mess)
            
            flags |= LOW_PSF_FLUX

        return flags

    def _run_model_fit(self, model):
        """
        wrapper to run fit, copy pars, maybe make plots
        """

        dindex=self.dindex
        psf_flux = self.data['psf_flux'][dindex,:].copy()
        psf_flux = psf_flux.clip(min=0.1, max=1.0e9)

        T = 2*(0.9/2.35)**2
        self.guesser=FromPSFGuesser(T, psf_flux)

        fitter=self._fit_model(self.mb_obs_list, model)

        self._copy_simple_pars(fitter)

        self._print_res(fitter)

        if self['make_plots']:
            self._do_make_plots(fitter, model)

        self.fitter=fitter


    def _fit_psf_flux(self):
        """
        Perform PSF flux fits on each band separately
        """

        dindex=self.dindex

        print('    fitting: psf')
        for band,obs_list in enumerate(self.mb_obs_list):
            self._fit_psf_flux_oneband(dindex, band, obs_list)

    def _fit_psf_flux_oneband(self, dindex, band, obs_list):
        """
        Fit the PSF flux in a single band
        """
        name='psf'

        fitter=ngmix.fitting.TemplateFluxFitter(obs_list, do_psf=True)
        fitter.go()

        res=fitter.get_result()
        data=self.data

        n=Namer(name)
        data[n('flags')][dindex,band] = res['flags']
        data[n('flux')][dindex,band] = res['flux']
        data[n('flux_err')][dindex,band] = res['flux_err']
        data[n('chi2per')][dindex,band] = res['chi2per']
        data[n('dof')][dindex,band] = res['dof']
        print("        %s flux(%s): %g +/- %g" % (name,band,res['flux'],res['flux_err']))

    def _get_multi_band_observations(self, mindex):
        """
        Get an ObsList object for the Coadd observations
        Get a MultiBandObsList object for the SE observations.
        """

        mb_obs_list=MultiBandObsList()

        for band in self.iband:

            self.band_wsum=0.0
            self.psfrec_counts_wsum=0.0

            obs = self._get_band_observation(band, mindex)
            obs_list = ObsList()
            obs_list.append( obs )

            mb_obs_list.append(obs_list)

        # means must go accross bands
        return mb_obs_list

    def _get_band_observation(self, band, mindex):
        """
        Get an Observation for a single band.

        GMixMaxIterEM is raised if psf fitting fails
        """
        import images
        meds=self.meds_list[band]

        icut=0
        fname = self._get_meds_orig_filename(meds, mindex, icut)
        im = self._get_meds_image(meds, mindex, icut)
        wt = self._get_meds_weight(meds, mindex, icut)
        jacob = self._get_jacobian(meds, mindex, icut)

        # for the psf fitting code
        wt=wt.clip(min=0.0)

        psf_obs = self._get_psf_observation(band, mindex, jacob)

        psf_fitter = self._fit_psf(psf_obs)
        psf_gmix = psf_fitter.get_gmix()

        # we only get here if psf fitting succeeds because an exception gets
        # raised on fit failure; note other metadata should always get set
        # above.  relies on global variable
        #
        # note this means that the psf counts sum and wsum should always
        # be incremented together

        psf_obs.set_gmix(psf_gmix)

        obs=Observation(im,
                        weight=wt,
                        jacobian=jacob,
                        psf=psf_obs)

        obs.filename=fname

        psf_fwhm=2.35*numpy.sqrt(psf_gmix.get_T()/2.0)
        print("        psf fwhm:",psf_fwhm)

        if self['make_plots']:
            self._do_make_psf_plots(band, psf_gmix, psf_obs, mindex, icut)

        return obs

    def _get_psf_observation(self, band, mindex, image_jacobian):
        """
        Get an Observation representing the PSF and the "sigma"
        from the psfex object
        """
        im, cen, sigma_pix, fname = self._get_psf_image(band, mindex)

        psf_jacobian = image_jacobian.copy()
        psf_jacobian.set_cen(cen[0], cen[1])

        psf_obs = Observation(im, jacobian=psf_jacobian)
        psf_obs.filename=fname

        # convert to sky coords
        sigma_sky = sigma_pix*psf_jacobian.get_scale()

        psf_obs.update_meta_data({'sigma_sky':sigma_sky})

        return psf_obs


    def _get_psf_image(self, band, mindex):
        """
        Get an image representing the psf
        """
        icut=0

        meds=self.meds_list[band]
        file_id=meds['file_id'][mindex,icut]

        pex=self.psfex_list[band]
        #print("    using psfex from:",pex['filename'])

        row=meds['orig_row'][mindex,icut]
        col=meds['orig_col'][mindex,icut]

        im=pex.get_rec(row,col).astype('f8', copy=False)
        cen=pex.get_center(row,col)
        sigma_pix=pex.get_sigma()

        return im, cen, sigma_pix, pex['filename']

    def _set_checkpoint_data(self):
        """
        See if checkpoint data was sent

        self._checkpoint_data should be set on construction
        """
        import fitsio
        if self.checkpoint_data is not None:
            self.data=self.checkpoint_data['data']

            # need the data to be native for the operation below
            fitsio.fitslib.array_to_native(self.data, inplace=True)

            # for nband==1 the written array drops the arrayness
            self.data.dtype=self._get_dtype()
 
    def _write_checkpoint(self, tm):
        """
        Write out the current data structure to a temporary
        checkpoint file.
        """
        import fitsio
        from .files import StagedOutFile

        print('checkpointing at',tm/60,'minutes')
        print(self.checkpoint_file)

        with StagedOutFile(self.checkpoint_file, tmpdir=self['work_dir']) as sf:
            with fitsio.FITS(sf.path,'rw',clobber=True) as fobj:
                fobj.write(self.data, extname="model_fits")


    def _get_dtype(self):
        self._check_models()

        nband=self['nband']
        bshape=(nband,)
        simple_npars=5+nband

        dt=[('id','i8'),
            ('number','i4'),
            ('processed','i1'),
            ('flags','i4'),
            ('time','f8'),

            ('box_size','i2')]


        # coadd fit with em 1 gauss
        # the psf flux fits are done for each band separately
        for name in ['psf']:
            n=Namer(name)
            dt += [(n('flags'),   'i4',bshape),
                   (n('flux'),    'f8',bshape),
                   (n('flux_err'),'f8',bshape),
                   (n('chi2per'),'f8',bshape),
                   (n('dof'),'f8',bshape)]

        if nband==1:
            cov_shape=(nband,)
        else:
            cov_shape=(nband,nband)

        models=self['fit_models']
        for model in models:

            n=Namer(model)

            np=simple_npars

            dt+=[(n('flags'),'i4'),
                 (n('pars'),'f8',np),
                 (n('pars_cov'),'f8',(np,np)),
                 (n('logpars'),'f8',np),
                 (n('logpars_cov'),'f8',(np,np)),
                 (n('flux'),'f8',bshape),
                 (n('flux_cov'),'f8',cov_shape),
                 (n('g'),'f8',2),
                 (n('g_cov'),'f8',(2,2)),
                
                 (n('s2n_w'),'f8'),
                 (n('chi2per'),'f8'),
                 (n('dof'),'f8'),
                 (n('aic'),'f8'),
                 (n('bic'),'f8'),
                 (n('arate'),'f8'),
                 (n('tau'),'f8'),
                ]
            if self['do_shear']:
                dt += [(n('g_sens'), 'f8', 2),
                       (n('P'), 'f8'),
                       (n('Q'), 'f8', 2),
                       (n('R'), 'f8', (2,2))]

        return dt

    def _make_struct(self):
        """
        make the output structure
        """
        dt=self._get_dtype()

        num=self.index_list.size
        data=numpy.zeros(num, dtype=dt)

        for name in ['psf']:
            n=Namer(name)
            data[n('flags')] = NO_ATTEMPT
            data[n('flux')] = DEFVAL
            data[n('flux_err')] = PDEFVAL
            data[n('chi2per')] = PDEFVAL

        for model in self['fit_models']:
            n=Namer(model)

            data[n('flags')] = NO_ATTEMPT

            data[n('pars')] = DEFVAL
            data[n('pars_cov')] = PDEFVAL
            data[n('flux')] = DEFVAL
            data[n('flux_cov')] = PDEFVAL
            data[n('g')] = DEFVAL
            data[n('g_cov')] = PDEFVAL

            data[n('s2n_w')] = DEFVAL
            data[n('chi2per')] = PDEFVAL
            data[n('aic')] = BIG_PDEFVAL
            data[n('bic')] = BIG_PDEFVAL

            data[n('tau')] = PDEFVAL

            if self['do_shear']:
                data[n('g_sens')] = DEFVAL
                data[n('P')] = DEFVAL
                data[n('Q')] = DEFVAL
                data[n('R')] = DEFVAL

     
        self.data=data



