"""
todo

    - copy coadd model fits as well
        - always do the fit and optionally do the ME guess from that

    - allow shear expand?  Only works for constant shear

    - use base 10 for logs?  I only used e based for the lognormal distributions

"""
from __future__ import print_function

import os
import time
import numpy
import meds
import psfex
import ngmix
from ngmix import srandu
from ngmix import Jacobian
from ngmix import GMixMaxIterEM, print_pars
from ngmix import Observation, ObsList, MultiBandObsList

from .lmfit import get_model_names

# starting new values for these
DEFVAL=-9999
PDEFVAL=9999
BIG_DEFVAL=-9.999e9
BIG_PDEFVAL=9.999e9


NO_SE_CUTOUTS=2**0
PSF_FIT_FAILURE=2**1
PSF_LARGE_OFFSETS=2**2
EXP_FIT_FAILURE=2**3
DEV_FIT_FAILURE=2**4

BOX_SIZE_TOO_BIG=2**5

EM_FIT_FAILURE=2**6

NO_ATTEMPT=2**30

#PSF_S2N=1.e6
PSF_OFFSET_MAX=0.25
PSF_TOL=1.0e-5
EM_MAX_TRY=3
EM_MAX_ITER=100

_CHECKPOINTS_DEFAULT_MINUTES=[30,60,110]

class MedsFit(object):
    def __init__(self, meds_files, **keys):
        """
        Model fitting

        parameters
        ----------
        meds_file:
            string of meds path, or list thereof for different bands

        The following are sent through keywords

        fit_types: list of strings
            ['simple']
        obj_range: optional
            a 2-element sequence or None.  If not None, only the objects in the
            specified range are processed and get_data() will only return those
            objects. The range is inclusive unlike slices.
        psf_offset_max: optional
            max offset between multi-component gaussians in psf models

        checkpoints: number, optional
            Times after which to checkpoint, seconds
        checkpoint_file: string, optional
            File which will hold a checkpoint.
        checkpoint_data: dict, optional
            The data representing a previous checkpoint, object and
            psf fits
        """

        self.conf={}
        self.conf.update(keys)

        self.meds_files=_get_as_list(meds_files)
        self.nband=len(self.meds_files)

        self.imstart=1
        self.fit_models=self.conf['fit_models']

        self.guess_type=keys['guess_type']

        self.nwalkers=keys['nwalkers']
        self.burnin=keys['burnin']
        self.nstep=keys['nstep']
        self.mca_a=keys.get('mca_a',2.0)

        self.do_shear=keys.get("do_shear",False)

        self._unpack_priors()


        self._setup_checkpoints()

        self.iband = range(self.nband)

        self._load_meds_files()
        self.psfex_lol = self._get_psfex_lol()

        self.obj_range=keys.get('obj_range',None)
        self._set_index_list()


        self.debug=keys.get('debug',0)

        self.psf_offset_max=keys.get('psf_offset_max',PSF_OFFSET_MAX)

        self.region=keys.get('region','seg_and_sky')
        self.max_box_size=keys.get('max_box_size',2048)

        self.reject_outliers=keys.get('reject_outliers',False) # from cutouts

        self.make_plots=keys.get('make_plots',False)
        self.prompt=keys.get('prompt',True)

        if self._checkpoint_data is None:
            self._make_struct()
            self._make_epoch_struct()

    def _unpack_priors(self):
        """
        Currently only separable priors
        """

        from ngmix.joint_prior import PriorSimpleSep
        from ngmix.priors import Disk2D
        conf=self.conf

        cen_prior=conf['cen_prior']

        g_prior_flat=Disk2D([0.0,0.0], 1.0)

        g_priors=conf['g_priors']
        T_priors=conf['T_priors']
        counts_priors=conf['counts_priors']

        models = self.fit_models
        nmod=len(models)

        nprior=len(g_priors)
        if nprior != nmod:
            raise ValueError("len(models)=%d but got len(priors)=%d" % (nmod,nprior))

        priors={}
        gflat_priors={}
        for i in xrange(nmod):
            model=self.fit_models[i]

            prior = PriorSimpleSep(cen_prior,
                                   g_priors[i],
                                   T_priors[i],
                                   counts_priors[i])

            # for the exploration, for which we do not apply g prior during
            gflat_prior = PriorSimpleSep(cen_prior,
                                         g_prior_flat,
                                         T_priors[i],
                                         counts_priors[i])

            priors[model]=prior
            gflat_priors[model]=gflat_prior

        self.priors=priors
        self.gflat_priors=gflat_priors

    def get_data(self):
        """
        Get the data structure.  If a subset was requested, only those rows are
        returned.
        """
        return self.data

    def get_epoch_data(self):
        """
        Get the epoch data structure, including psf fitting
        """
        return self.epoch_data

    def get_meds_meta_list(self):
        """
        get copies of the meta data
        """
        return [m.copy() for m in self.meds_meta_list]

    def get_magzp(self):
        """
        Get the magnitude zero point.
        """
        return self.meds_meta['magzp_ref'][0]

    def do_fits(self):
        """
        Fit all objects in our list
        """

        t0=time.time()

        last=self.index_list[-1]
        num=len(self.index_list)

        for dindex in xrange(num):
            if self.data['processed'][dindex]==1:
                # was checkpointed
                continue

            mindex = self.index_list[dindex]
            print('index: %d:%d' % (mindex,last), )
            self.fit_obj(dindex)

            tm=time.time()-t0

            self._try_checkpoint(tm) # only at certain intervals

        tm=time.time()-t0
        print("time:",tm)
        print("time per:",tm/num)


    def fit_obj(self, dindex):
        """
        Process the indicated object through the requested fits
        """

        t0=time.time()

        # for checkpointing
        self.data['processed'][dindex]=1

        mindex = self.index_list[dindex]

        ncutout_tot=self._get_object_ncutout(mindex)
        self.data['nimage_tot'][dindex, :] = ncutout_tot

        # need to do this because we work on subset files
        self.data['number'][dindex] = self.meds_list[0]['number'][mindex]

        flags = self._obj_check(mindex)
        if flags != 0:
            self.data['flags'][dindex] = flags
            return 0

        # MultiBandObsList obects
        coadd_mb_obs_list, mb_obs_list, n_im = \
                self._get_multi_band_observations(mindex)

        if (len(coadd_mb_obs_list) == 0 and self.guess_from_coadd):
            print("  Coadd psf fitting failed, Could not guess from coadd")
            self.data['flags'][dindex] = PSF_FIT_FAILURE 
            return
        if len(mb_obs_list) == 0:
            print("  psf fitting failed")
            self.data['flags'][dindex] = PSF_FIT_FAILURE 
            return

        print(mb_obs_list[0][0].image.shape)

        self.data['nimage_use'][dindex, :] = n_im

        sdata={'coadd_mb_obs_list':coadd_mb_obs_list,
               'mb_obs_list':mb_obs_list}

        flags=self._fit_all_models(dindex, sdata)
        self.data['flags'][dindex] = flags

        self.data['time'][dindex] = time.time()-t0

    def _get_object_ncutout(self, mindex):
        """
        number of cutouts for the specified object.
        """
        ncutout=0
        for meds in self.meds_list:
            ncutout += meds['ncutout'][mindex]
        return ncutout

    def _obj_check(self, mindex):
        """
        Check box sizes, number of cutouts
        """
        for band in self.iband:
            meds=self.meds_list[band]
            flags=self._obj_check_one(meds, mindex)
            if flags != 0:
                break
        return flags

    def _obj_check_one(self, meds, mindex):
        """
        Check box sizes, number of cutouts
        """
        flags=0

        box_size=meds['box_size'][mindex]
        if box_size > self.max_box_size:
            print('Box size too big:',box_size)
            flags |= BOX_SIZE_TOO_BIG

        if meds['ncutout'][mindex] < 2:
            print('No SE cutouts')
            flags |= NO_SE_CUTOUTS
        return flags



    def _get_multi_band_observations(self, mindex):
        """
        Get an ObsList object for the Coadd observations
        Get a MultiBandObsList object for the SE observations.
        """

        coadd_mb_obs_list=MultiBandObsList()
        mb_obs_list=MultiBandObsList()

        # number used
        n_im = 0
        for band in self.iband:
            cobs_list, obs_list = self._get_band_observations(band, mindex)

            if len(cobs_list) > 0:
                coadd_mb_obs_list.append(cobs_list)

            this_n_im=len(obs_list)
            if this_n_im > 0:
                mb_obs_list.append(obs_list)
            n_im += this_n_im

        return coadd_mb_obs_list, mb_obs_list, n_im

    def _get_band_observations(self, band, mindex):
        """
        Get an ObsList for the coadd observations in each band

        If psf fitting fails, the ObsList will be zero length
        """

        meds=self.meds_list[band]
        ncutout=meds['ncutout'][mindex]

        coadd_obs_list=ObsList()
        obs_list = ObsList()

        icut=0
        try:
            coadd_obs = self._get_band_observation(band, mindex, icut)
            coadd_obs_list.append( coadd_obs )
            flags=0
        except GMixMaxIterEM:
            flags=PSF_FIT_FAILURE
        self._set_psf_meta(meds, mindex, band, icut, flags)
        self.psf_index += 1

        for icut in xrange(1,ncutout):
            try:
                obs = self._get_band_observation(band, mindex, icut)
                obs_list.append(obs)
                flags=0
            except GMixMaxIterEM:
                flags=PSF_FIT_FAILURE

            # we set the metadata even if the fit fails
            self._set_psf_meta(meds, mindex, band, icut, flags)
            self.psf_index += 1
        
        return coadd_obs_list, obs_list

    def _get_band_observation(self, band, mindex, icut):
        """
        Get an Observation for a single band.

        GMixMaxIterEM is raised if psf fitting fails
        """
        import images
        meds=self.meds_list[band]

        im = self._get_meds_image(meds, mindex, icut)
        wt = self._get_meds_weight(meds, mindex, icut)
        jacob = self._get_jacobian(meds, mindex, icut)

        psf_obs = self._get_psf_observation(band, mindex, icut, jacob)

        psf_fitter = self._fit_psf(psf_obs)
        psf_gmix = psf_fitter.get_gmix()

        # the psf fit only gets set if we succeed; not other metadata should
        # always get set above.  relies on global variable self.psf_index
        self._set_psf_result(psf_gmix)

        psf_obs.set_gmix(psf_gmix)

        obs=Observation(im,
                        weight=wt,
                        jacobian=jacob,
                        psf=psf_obs)

        psf_fwhm=2.35*numpy.sqrt(psf_gmix.get_T()/2.0)
        print("        psf fwhm:",psf_fwhm)

        return obs

    def _fit_psf(self, obs):
        """
        Fit the PSF observation to a gaussian mixture

        If no fit after psf_ntry tries, GMixMaxIterEM will
        be raised

        """

        # already in sky coordinates
        sigma_guess=obs.meta['sigma_sky']

        empars=self.conf['psf_em_pars']
        fitter = self._fit_with_em(obs,
                                   sigma_guess,
                                   empars['ngauss'],
                                   empars['maxiter'],
                                   empars['tol'],
                                   empars['ntry'])

        return fitter


    def _get_jacobian(self, meds, mindex, icut):
        """
        Get a Jacobian object for the requested object
        """
        jdict = meds.get_jacobian(mindex, icut)
        jacob = self._convert_jacobian_dict(jdict)
        return jacob

    def _get_psf_observation(self, band, mindex, icut, image_jacobian):
        """
        Get an Observation representing the PSF and the "sigma"
        from the psfex object
        """
        im, cen, sigma_pix = self._get_psf_image(band, mindex, icut)

        psf_jacobian = image_jacobian.copy()
        psf_jacobian.set_cen(cen[0], cen[1])

        psf_obs = Observation(im, jacobian=psf_jacobian)

        # convert to sky coords
        sigma_sky = sigma_pix*psf_jacobian.get_scale()

        psf_obs.update_meta_data({'sigma_sky':sigma_sky})

        return psf_obs

    def _get_psf_image(self, band, mindex, icut):
        """
        Get an image representing the psf
        """

        meds=self.meds_list[band]
        file_id=meds['file_id'][mindex,icut]

        pex=self.psfex_lol[band][file_id]

        row=meds['orig_row'][mindex,icut]
        col=meds['orig_col'][mindex,icut]

        im=pex.get_rec(row,col).astype('f8', copy=False)
        cen=pex.get_center(row,col)
        sigma_pix=pex.get_sigma()

        return im, cen, sigma_pix

    def _get_meds_image(self, meds, mindex, icut):
        """
        Get an image cutout from the input MEDS file
        """
        im = meds.get_cutout(mindex, icut)
        im = im.astype('f8', copy=False)
        return im

    def _get_meds_weight(self, meds, mindex, icut):
        """
        Get a weight map from the input MEDS file
        """
        if self.region=='seg_and_sky':
            wt=meds.get_cweight_cutout(mindex, icut)
        elif self.region=="cweight-nearest":
            wt=meds.get_cweight_cutout_nearest(mindex, icut)
        else:
            raise ValueError("support other region types")

        wt=wt.astype('f8', copy=False)
        return wt

    def _convert_jacobian_dict(self, jdict):
        """
        Get the jacobian for the input meds index and cutout index
        """
        jacob=Jacobian(jdict['row0'],
                       jdict['col0'],
                       jdict['dudrow'],
                       jdict['dudcol'],
                       jdict['dvdrow'],
                       jdict['dvdcol'])
        return jacob



    def _set_psf_result(self, gm):
        """
        Set psf fit data.
        """

        psf_index=self.psf_index

        pars=gm.get_full_pars()
        g1,g2,T=gm.get_g1g2T()

        ed=self.epoch_data
        ed['psf_fit_g'][psf_index,0]    = g1
        ed['psf_fit_g'][psf_index,1]    = g2
        ed['psf_fit_T'][psf_index]      = T
        ed['psf_fit_pars'][psf_index,:] = pars

    def _set_psf_meta(self, meds, mindex, band, icut, flags):
        """
        Set all the meta data for the psf result
        """
        psf_index=self.psf_index
        ed=self.epoch_data

        # mindex can be an index into a sub-range meds
        ed['number'][psf_index] = meds['number'][mindex]
        ed['band_num'][psf_index] = band
        ed['cutout_index'][psf_index] = icut
        ed['file_id'][psf_index]  = meds['file_id'][mindex,icut].astype('i4')
        ed['orig_row'][psf_index] = meds['orig_row'][mindex,icut]
        ed['orig_col'][psf_index] = meds['orig_col'][mindex,icut]
        ed['psf_fit_flags'][psf_index] = flags


    def _should_keep_psf(self, gm):
        """
        For double gauss we limit the separation
        """
        keep=True
        offset_arcsec=0.0
        psf_ngauss=self.conf['psf_em_pars']['ngauss']
        if psf_ngauss == 2:
            offset_arcsec = calc_offset_arcsec(gm)
            if offset_arcsec > self.psf_offset_max:
                keep=False

        return keep, offset_arcsec


    def _fit_all_models(self, dindex, sdata):
        """
        Fit psf flux and other models
        """

        flags=self._fit_coadd_em1(dindex, sdata)
        if flags != 0:
            print("    failure fitting coadd em1")
            return flags

        # this can fail and we continue
        self._fit_psf_flux(dindex, sdata)

        s2n=self.data['coadd_em1_flux'][dindex,:]/self.data['coadd_em1_flux_err'][dindex,:]
        max_s2n=numpy.nanmax(s2n)
         
        if max_s2n >= self.conf['min_em1_s2n']:
            for model in self.fit_models:
                print('    fitting:',model)

                self._run_model_fit(dindex,
                                    sdata,
                                    model,
                                    coadd=True)
                self._run_model_fit(dindex,
                                    sdata,
                                    model,
                                    coadd=False)
        else:
            mess="    em1 s/n too low: %s (%s)"
            mess=mess % (max_psf_s2n,self.conf['min_em1_s2n'])
            print(mess)

        return flags


    def _fit_coadd_em1(self, dindex, sdata):
        """
        Fit a the multi-band galaxy observations to one gaussian using EM

        This should be the coadd or some other single-observation-per-band
        data
        """

        print('    fitting: coadd em1')
        mb_obs_list=sdata['coadd_mb_obs_list']

        all_flags=0
        for band in self.iband:
            obs = mb_obs_list[band][0]
            gmix,flags=self._fit_coadd_em1_oneband(dindex, band, obs)

            if flags == 0:
                obs.set_gmix(gmix)
                flags += self._fit_coadd_template_flux_oneband(dindex,
                                                               band,
                                                               obs)
            all_flags += flags

        pars=self.data['coadd_em1_gmix_pars'][dindex,:]
        print_pars(pars, front='        coadd em1:')
        return all_flags

    def _fit_coadd_em1_oneband(self, dindex, band, obs):
        """
        Fit a single observation to a single gaussian
        """
        psf_gmix = obs.psf.gmix

        # this is in sky coords
        sigma_guess = numpy.sqrt( psf_gmix.get_T()/2.0 )
        #print('    galaxy sigma guess:',sigma_guess)

        ngauss=1
        empars=self.conf['galaxy_em_pars']

        flags=0
        try:
            fitter = self._fit_with_em(obs,
                                       sigma_guess,
                                       ngauss,
                                       empars['maxiter'],
                                       empars['tol'],
                                       empars['ntry'])
        except GMixMaxIterEM:
            fitter=None
            flags=EM_FIT_FAILURE

        data=self.data
        data['coadd_em1_flags'][dindex,band] = flags

        if flags == 0:
            gmix=fitter.get_gmix()
            pars=gmix.get_full_pars()
            #print(pars)
            beg=band*6
            end=(band+1)*6
            data['coadd_em1_gmix_pars'][dindex,beg:end] = pars
        else:
            gmix=None

        return gmix, flags


    def _fit_coadd_template_flux_oneband(self, dindex, band, obs):
        """
        Use the gmix as a template and fit for the flux. You must
        have run obs.set_gmix() and that gmix center should be the
        best one from the em fit

        We take the centers, relative to the jacobian centers which must
        be co-located, from the gmix themselves
        """
        fitter = ngmix.fitting.TemplateFluxFitter(obs)
        fitter.go()

        res=fitter.get_result()
        flags=res['flags']
        data=self.data
        if flags == 0:
            data['coadd_em1_flux'][dindex,band] = res['flux']
            data['coadd_em1_flux_err'][dindex,band] = res['flux_err']
            data['coadd_em1_chi2per'][dindex,band] = res['chi2per']
            data['coadd_em1_dof'][dindex,band] = res['dof']
            print("        coadd em1 flux(%s): %g +/- %g" % (band,res['flux'],res['flux_err']))
        else:
            print("        could not fit template for band:",band)

        return flags


    def _fit_with_em(self, obs, sigma_guess, ngauss, maxiter, tol, ntry):
        """
        Fit a gaussian mixture to the input observation using em to find a
        """

        s2guess = sigma_guess**2

        im_with_sky, sky = ngmix.em.prep_image(obs.image)

        tobs = Observation(im_with_sky, jacobian=obs.jacobian)
        fitter=ngmix.em.GMixEM(tobs)

        for i in xrange(ntry):

            gm_guess=self._get_em_guess(s2guess, ngauss)

            try:
                fitter.go(gm_guess,
                          sky,
                          maxiter=maxiter,
                          tol=tol)
                break
            except GMixMaxIterEM:
                res=fitter.get_result()
                print('last fit:')
                print( fitter.get_gmix() )
                print( 'try:',i+1,'fdiff:',res['fdiff'],'numiter:',res['numiter'] )
                if i == (ntry-1):
                    raise

        return fitter




    def _get_em_guess(self, sigma2, ngauss):
        """
        Guess for the EM algorithm
        """

        if ngauss==1:
            pars=numpy.array( [1.0, 0.0, 0.0, 
                               sigma2*(1.0 + 0.1*srandu()),
                               0.0,
                               sigma2*(1.0 + 0.1*srandu())] )
        elif ngauss==2:

            pars=numpy.array( [_em2_pguess[0],
                               0.1*srandu(),
                               0.1*srandu(),
                               _em2_fguess[0]*sigma2*(1.0 + 0.1*srandu()),
                               0.0,
                               _em2_fguess[0]*sigma2*(1.0 + 0.1*srandu()),

                               _em2_pguess[1],
                               0.1*srandu(),
                               0.1*srandu(),
                               _em2_fguess[1]*sigma2*(1.0 + 0.1*srandu()),
                               0.0,
                               _em2_fguess[1]*sigma2*(1.0 + 0.1*srandu())] )
        elif ngauss==3:

            pars=numpy.array( [_em3_pguess[0]*(1.0+0.1*srandu()),
                               0.1*srandu(),
                               0.1*srandu(),
                               _em3_fguess[0]*sigma2*(1.0 + 0.1*srandu()),
                               0.01*srandu(),
                               _em3_fguess[0]*sigma2*(1.0 + 0.1*srandu()),

                               _em3_pguess[1]*(1.0+0.1*srandu()),
                               0.1*srandu(),
                               0.1*srandu(),
                               _em3_fguess[1]*sigma2*(1.0 + 0.1*srandu()),
                               0.01*srandu(),
                               _em3_fguess[1]*sigma2*(1.0 + 0.1*srandu()),

                               _em3_pguess[2]*(1.0+0.1*srandu()),
                               0.1*srandu(),
                               0.1*srandu(),
                               _em3_fguess[2]*sigma2*(1.0 + 0.1*srandu()),
                               0.01*srandu(),
                               _em3_fguess[2]*sigma2*(1.0 + 0.1*srandu())]

                            )

        else:
            raise ValueError("only support 1,2,3 gauss for em")

        return ngmix.gmix.GMix(pars=pars)




    def _fit_psf_flux(self, dindex, sdata):
        """
        Perform PSF flux fits on each band separately
        """

        print('    fitting: psf')
        for band,obs_list in enumerate(sdata['mb_obs_list']):
            self._fit_psf_flux_oneband(dindex, band, obs_list)

        #print_pars(self.data['psf_flux'][dindex],     front='        ')
        #print_pars(self.data['psf_flux_err'][dindex], front='        ')


    def _fit_psf_flux_oneband(self, dindex, band, obs_list):
        """
        Fit the PSF flux in a single band
        """
        fitter=ngmix.fitting.TemplateFluxFitter(obs_list, do_psf=True)
        fitter.go()

        res=fitter.get_result()
        data=self.data

        data['psf_flags'][dindex,band] = res['flags']
        data['psf_flux'][dindex,band] = res['flux']
        data['psf_flux_err'][dindex,band] = res['flux_err']
        data['psf_chi2per'][dindex,band] = res['chi2per']
        data['psf_dof'][dindex,band] = res['dof']
        print("        psf flux(%s): %g +/- %g" % (band,res['flux'],res['flux_err']))



    def _run_model_fit(self, dindex, sdata, model, coadd=False):
        """
        wrapper to run fit, copy pars, maybe make plots

        sets .fitter or .coadd_fitter
        """
        if coadd:
            guess_type='em1'
            mb_obs_list=sdata['coadd_mb_obs_list']
        else:
            guess_type=self.guess_type
            mb_obs_list=sdata['mb_obs_list']

        fitter=self._fit_model(dindex,
                               mb_obs_list,
                               model,
                               guess_type=guess_type)

        self._copy_simple_pars(dindex, fitter, coadd=coadd)

        lin_res=fitter.get_lin_result()
        self._print_res(lin_res, coadd=coadd)

        if self.make_plots:
            self._do_make_plots(dindex, fitter, model, coadd=coadd)

        if coadd:
            self.coadd_fitter=fitter
        else:
            self.fitter=fitter

    def _fit_model(self, dindex, mb_obs_list, model, guess_type='em1'):
        """
        Fit all the simple models
        """

        guess=self._get_guess(dindex, guess_type)

        fitter=self._fit_simple(dindex, mb_obs_list, model, guess)

        # also adds .weights attribute
        self._calc_mcmc_stats(fitter, model)

        log_res=fitter.get_result()
        lin_res=fitter.get_lin_result()

        if self.do_shear:
            self._add_shear_info(log_res, fitter, model)

        return fitter

    def _get_guess(self, dindex, guess_type):
        """
        get a guess
        """
        if guess_type=='coadd':
            guess=self._get_guess_from_coadd()
        elif guess_type=='em1':
            guess=self._get_guess_from_em1(dindex)
        else:
            raise ValueError("bad guess type: '%s'" % guess_type)

        return guess



    def _fit_simple(self, dindex, mb_obs_list, model, guess):
        """
        Fit one of the "simple" models, e.g. exp or dev

        use flat g prior
        """

        from ngmix.fitting import MCMCSimple

        prior=self.gflat_priors[model]

        fitter=MCMCSimple(mb_obs_list,
                          model,
                          prior=prior,
                          nwalkers=self.nwalkers,
                          mca_a=self.mca_a)

        pos=fitter.run_mcmc(guess,self.burnin)
        pos=fitter.run_mcmc(pos,self.nstep)

        return fitter


    def _get_guess_from_em1(self, dindex):
        """
        take guesses from em1 except for center and ellipticity.

        The size will be too big due to the psf...

        Don't take center from em1 because we don't use the mask for em1 and
        the center may well drift.

        """
        from numpy import log10

        data=self.data

        gmix_pars=data['coadd_em1_gmix_pars'][dindex,:]

        nband=self.nband
        ind=numpy.arange(nband)

        irr=gmix_pars[3+ind*6]
        icc=gmix_pars[5+ind*6]
        T = numpy.median( irr+icc ).clip(min=0.1, max=100.0)


        flux_vals=data['coadd_em1_flux'][dindex,:].clip(min=0.1, max=1.0e6)

        print("        em1 guess:",T,flux_vals)

        nwalkers=self.nwalkers
        np=5+self.nband

        guess=numpy.zeros( (nwalkers, np) )
        guess[:,0] = 0.01*srandu(nwalkers)
        guess[:,1] = 0.01*srandu(nwalkers)
        guess[:,2] = 0.1*srandu(nwalkers)
        guess[:,3] = 0.1*srandu(nwalkers)
        guess[:,4] = log10( T*(1.0 + 0.1*srandu(nwalkers)) )
        for band in self.iband:
            guess[:,5+band] = log10( flux_vals[band]*(1.0 + 0.1*srandu(nwalkers)) )

        return guess

    def _get_guess_from_coadd(self):
        """
        get a random set of points from the coadd chain
        """
        import random

        print('        getting guess from coadd')

        # get a random set (the most recent would be from the same walker)
        log_trials = self.coadd_fitter.get_trials()
        np = log_trials.shape[0]
        rand_int = random.sample(xrange(np), self.nwalkers)
        return log_trials[rand_int, :]

    def _calc_mcmc_stats(self, fitter, model):
        """
        Add some stats for the mcmc chain

        Also add a weights attribute to the fitter
        """
        log_trials=fitter.get_trials()

        g_prior=self.priors[model].g_prior
        weights = g_prior.get_prob_array2d(log_trials[:,2], log_trials[:,3])
        fitter.calc_result(weights=weights)
        fitter.calc_lin_result(weights=weights)

        fitter.weights=weights

    def _add_shear_info(self, res, fitter, model):
        """
        Add pqr or lensfit info
        """

        trials=fitter.get_trials()
        g=trials[:,2:2+2]

        g_prior=self.priors[model].g_prior

        ls=ngmix.lensfit.LensfitSensitivity(g, g_prior)
        res['g_sens'] = ls.get_g_sens()
        res['nuse'] = ls.get_nuse()

        pqrobj=ngmix.pqr.PQR(g, g_prior)
        P,Q,R = pqrobj.get_pqr()
        res['P']=P
        res['Q']=Q
        res['R']=R



    def _copy_simple_pars(self, dindex, fitter, coadd=False):
        """
        Copy from the result dict to the output array
        """

        log_res=fitter.get_result()
        lin_res=fitter.get_lin_result()

        model=log_res['model']
        if coadd:
            model = 'coadd_%s' % model

        n=get_model_names(model)

        self.data[n['flags']][dindex] = log_res['flags']

        if log_res['flags'] == 0:
            log_pars=log_res['pars']
            log_pars_cov=log_res['pars_cov']
            lin_pars=log_res['pars']
            lin_pars_cov=log_res['pars_cov']

            flux=lin_pars[5:]
            flux_cov=lin_pars_cov[5:, 5:]

            self.data[n['pars']][dindex,:] = lin_pars
            self.data[n['pars_cov']][dindex,:,:] = lin_pars_cov
            self.data[n['logpars']][dindex,:] = log_pars
            self.data[n['logpars_cov']][dindex,:,:] = log_pars_cov


            self.data[n['flux']][dindex] = flux
            self.data[n['flux_cov']][dindex] = flux_cov

            self.data[n['g']][dindex,:] = log_res['g']
            self.data[n['g_cov']][dindex,:,:] = log_res['g_cov']

            self.data[n['arate']][dindex] = log_res['arate']
            if log_res['tau'] is not None:
                self.data[n['tau']][dindex] = log_res['tau']

            for sn in _stat_names:
                self.data[n[sn]][dindex] = log_res[sn]

            if self.do_shear:
                self.data[n['g_sens']][dindex,:] = log_res['g_sens']
                self.data[n['P']][dindex] = log_res['P']
                self.data[n['Q']][dindex,:] = log_res['Q']
                self.data[n['R']][dindex,:,:] = log_res['R']
                

    def _do_make_plots(self, dindex, fitter, model, coadd=False):
        """
        make plots
        """

        if coadd:
            type='coadd'
        else:
            type='mb'

        mindex = self.index_list[dindex]
        pdict=fitter.make_plots(title='%s %s' % (type,model),
                                weights=fitter.weights,
                                do_residual=True)

        trials_png='%s-trials-%06d-%s.png' % (type,mindex,model)
        wtrials_png='%s-wtrials-%06d-%s.png' % (type,mindex,model)

        print("            ",trials_png)
        pdict['trials'].write_img(1200,1200,trials_png)

        print("            ",wtrials_png)
        pdict['wtrials'].write_img(1200,1200,wtrials_png)

        for band, band_plots in enumerate(pdict['resid']):
            for icut, plt in enumerate(band_plots):
                fname='%s-resid-%06d-%s-band%d-im%d.png' % (type,mindex,model,band,icut+1)
                print("            ",fname)
                plt.write_img(1920,1200,fname)


    def _load_meds_files(self):
        """
        Load all listed meds files
        """

        self.meds_list=[]
        self.meds_meta_list=[]

        for i,f in enumerate(self.meds_files):
            print(f)
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

        self.nobj_tot = self.meds_list[0].size

    def _get_psfex_lol(self):
        """
        Load psfex objects for each of the SE images
        include the coadd so we get  the index right
        """
        print('loading psfex')
        desdata=os.environ['DESDATA']
        meds_desdata=self.meds_list[0]._meta['DESDATA'][0]

        psfex_lol=[]

        for band in self.iband:
            meds=self.meds_list[band]

            psfex_list = self._get_psfex_objects(meds)
            psfex_lol.append( psfex_list )

        return psfex_lol

    def _get_psfex_objects(self, meds):
        """
        Load psfex objects for each of the SE images
        include the coadd so we get  the index right
        """
        desdata=os.environ['DESDATA']
        meds_desdata=meds._meta['DESDATA'][0]

        psfex_list=[]
        info=meds.get_image_info()
        nimage=info.size

        for i in xrange(nimage):
            impath=info['image_path'][i].strip()
            psfpath=impath.replace('.fits.fz','_psfcat.psf')

            if desdata not in psfpath:
                psfpath=psfpath.replace(meds_desdata,desdata)

            if not os.path.exists(psfpath):
                raise IOError("missing psfex: %s" % psfpath)
            else:
                pex=psfex.PSFEx(psfpath)
            psfex_list.append(pex)

        return psfex_list

    def _set_index_list(self):
        """
        set the list of indices to be processed
        """
        if self.obj_range is None:
            start=0
            end=self.nobj_tot-1
        else:
            start=self.obj_range[0]
            end=self.obj_range[1]

        self.index_list = numpy.arange(start,end+1)


    def _print_res(self, res, coadd=False):
        if res['flags']==0:
            if coadd:
                type='coadd'
            else:
                type='mb'
            print("        %s linear pars:" % type)
            print_pars(res['pars'],    front='        ')
            print_pars(res['pars_err'],front='        ')
            print('        arate:',res['arate'])

    '''
    def _print_simple_shape(self, res):
        g1=res['pars'][2]
        g1err=numpy.sqrt(res['pars_cov'][2,2])
        g2=res['pars'][3]
        g2err=numpy.sqrt(res['pars_cov'][3,3])

        print('        g1: %.4g +/- %.4g g2: %.4g +/- %.4g' % (g1,g1err,g2,g2err) )

    def _print_simple_fluxes(self, res):
        """
        print in a nice format
        """
        from numpy import sqrt,diag
        flux=res['pars'][5:]
        flux_cov=res['pars_cov'][5:, 5:]
        flux_err=sqrt(diag(flux_cov))

        print_pars(flux,     front='        flux')
        print_pars(flux_err, front='        ferr')

    def _print_simple_T(self, res):
        """
        print T, Terr, Ts2n and sigma
        """
        T = res['pars'][4]
        Terr = numpy.sqrt( res['pars_cov'][4,4] )

        if Terr > 0:
            Ts2n=T/Terr
        else:
            Ts2n=-9999.0
        if T > 0:
            sigma=numpy.sqrt(T/2.)
        else:
            sigma=-9999.0

        tup=(T,Terr,Ts2n,sigma)
        print('        T: %s +/- %s Ts2n: %s sigma: %s' % tup )
    '''

    def _setup_checkpoints(self):
        """
        Set up the checkpoint times in minutes and data
        """
        self.checkpoints = self.conf.get('checkpoints',_CHECKPOINTS_DEFAULT_MINUTES)
        self.n_checkpoint    = len(self.checkpoints)
        self.checkpointed    = [0]*self.n_checkpoint
        self.checkpoint_file = self.conf.get('checkpoint_file',None)

        self._set_checkpoint_data()

        if self.checkpoint_file is not None:
            self.do_checkpoint=True
        else:
            self.do_checkpoint=False

    def _set_checkpoint_data(self):
        """
        See if checkpoint data was sent
        """
        self._checkpoint_data=self.conf.get('checkpoint_data',None)
        if self._checkpoint_data is not None:
            self.data=self._checkpoint_data['data']

            # for nband==1 the written array drops the arrayness
            self.data.dtype=self._get_dtype()
            self.epoch_data=self._checkpoint_data['epoch_data']

            if self.epoch_data.dtype.names is not None:
                # start where we left off
                w,=numpy.where( self.epoch_data['number'] == -1)
                self.psf_index = w.min()

    def _try_checkpoint(self, tm):
        """
        Checkpoint at certain intervals.  
        Potentially modified self.checkpointed
        """

        should_checkpoint, icheck = self._should_checkpoint(tm)

        if should_checkpoint:
            self._write_checkpoint(tm)
            self.checkpointed[icheck]=1

    def _should_checkpoint(self, tm):
        """
        Should we write a checkpoint file?
        """

        should_checkpoint=False
        icheck=-1

        if self.do_checkpoint:
            tm_minutes=tm/60

            for i in xrange(self.n_checkpoint):

                checkpoint=self.checkpoints[i]
                checkpointed=self.checkpointed[i]

                if tm_minutes > checkpoint and not checkpointed:
                    should_checkpoint=True
                    icheck=i

        return should_checkpoint, icheck

    def _write_checkpoint(self, tm):
        """
        Write out the current data structure to a temporary
        checkpoint file.
        """
        import fitsio

        print('checkpointing at',tm/60,'minutes')
        print(self.checkpoint_file)

        with fitsio.FITS(self.checkpoint_file,'rw',clobber=True) as fobj:
            fobj.write(self.data, extname="model_fits")
            fobj.write(self.epoch_data, extname="epoch_data")

    def _check_models(self):
        """
        make sure all models are supported
        """
        for model in self.fit_models:
            if model not in ['exp','dev']:
                raise ValueError("model '%s' not supported" % model)

    def _get_all_models(self):
        """
        get all model names, includeing the coadd_ ones
        """
        coadd_models=['coadd_%s' % model for model in self.fit_models]
        models = coadd_models + self.fit_models

        return models

    def _count_all_cutouts(self):
        """
        Count the cutouts for the objects, including the coadd, space which may
        not be used.  If obj_range was sent, this will be a subset
        """
        ncutout=0
        ncoadd=self.index_list.size
        for meds in self.meds_list:
            ncutout += meds['ncutout'][self.index_list].sum()
        return ncutout


    def _make_epoch_struct(self):
        """
        We will make the maximum number of possible psfs according
        to the cutout count, not counting the coadd
        """

        psf_ngauss=self.conf['psf_em_pars']['ngauss']
        npars=psf_ngauss*6
        dt=[('number','i4'), # 1-n as in sextractor
            ('band_num','i2'),
            ('cutout_index','i4'), # this is the index into e.g. m['orig_row'][3,index]
            ('orig_row','f8'),
            ('orig_col','f8'),
            ('file_id','i4'),
            ('psf_fit_flags','i4'),
            ('psf_fit_g','f8',2),
            ('psf_fit_T','f8'),
            ('psf_fit_pars','f8',npars)]

        ncutout=self._count_all_cutouts()
        if ncutout > 0:
            epoch_data = numpy.zeros(ncutout, dtype=dt)

            epoch_data['number'] = -1
            epoch_data['band_num'] = -1
            epoch_data['cutout_index'] = -1
            epoch_data['file_id'] = -1
            epoch_data['psf_fit_g'] = PDEFVAL
            epoch_data['psf_fit_T'] = PDEFVAL
            epoch_data['psf_fit_pars'] = PDEFVAL
            epoch_data['psf_fit_flags'] = NO_ATTEMPT

            self.epoch_data=epoch_data
        else:
            self.epoch_data=numpy.zeros(1)

        # where the next psf data will be written
        self.psf_index = 0


    def _get_dtype(self):
        self._check_models()

        nband=self.nband
        bshape=(nband,)
        simple_npars=5+nband

        dt=[('number','i4'),
            ('processed','i1'),
            ('flags','i4'),
            ('nimage_tot','i4',bshape),
            ('nimage_use','i4',bshape),
            ('time','f8')]

        # coadd fit with em 1 gauss
        n=get_model_names('coadd_em1')
        dt += [(n['flags'],'i4',bshape),
               (n['gmix_pars'],'f8',6*nband),
               (n['flux'], 'f8', bshape),
               (n['flux_err'], 'f8', bshape),
               (n['chi2per'],'f8',bshape),
               (n['dof'],'f8',bshape)]


        # the psf flux fits are done for each band separately
        n=get_model_names('psf')
        dt += [(n['flags'],   'i4',bshape),
               (n['flux'],    'f8',bshape),
               (n['flux_err'],'f8',bshape),
               (n['chi2per'],'f8',bshape),
               (n['dof'],'f8',bshape)]

        models=self._get_all_models()
        for model in models:

            if nband==1:
                cov_shape=(nband,)
            else:
                cov_shape=(nband,nband)

            n=get_model_names(model)

            np=simple_npars

            dt+=[(n['flags'],'i4'),
                 (n['pars'],'f8',np),
                 (n['pars_cov'],'f8',(np,np)),
                 (n['logpars'],'f8',np),
                 (n['logpars_cov'],'f8',(np,np)),
                 (n['flux'],'f8',bshape),
                 (n['flux_cov'],'f8',cov_shape),
                 (n['g'],'f8',2),
                 (n['g_cov'],'f8',(2,2)),
                
                 (n['s2n_w'],'f8'),
                 (n['chi2per'],'f8'),
                 (n['dof'],'f8'),
                 (n['aic'],'f8'),
                 (n['bic'],'f8'),
                 (n['arate'],'f8'),
                 (n['tau'],'f8'),
                ]
            if self.do_shear:
                dt += [(n['g_sens'], 'f8', 2),
                       (n['P'], 'f8'),
                       (n['Q'], 'f8', 2),
                       (n['R'], 'f8', (2,2))]

        return dt

    def _make_struct(self):
        """
        make the output structure
        """
        dt=self._get_dtype()

        num=self.index_list.size
        data=numpy.zeros(num, dtype=dt)

        n=get_model_names('psf')
        data[n['flags']] = NO_ATTEMPT
        data[n['flux']] = DEFVAL
        data[n['flux_err']] = PDEFVAL
        data[n['chi2per']] = PDEFVAL

        n=get_model_names('coadd_em1')
        data[n['flags']] = NO_ATTEMPT
        data[n['gmix_pars']] = DEFVAL
        data[n['flux']] = DEFVAL
        data[n['flux_err']] = PDEFVAL
        data[n['chi2per']] = PDEFVAL

        for model in self.fit_models:
            n=get_model_names(model)

            data[n['flags']] = NO_ATTEMPT

            data[n['pars']] = DEFVAL
            data[n['pars_cov']] = PDEFVAL
            data[n['flux']] = DEFVAL
            data[n['flux_cov']] = PDEFVAL
            data[n['g']] = DEFVAL
            data[n['g_cov']] = PDEFVAL

            data[n['s2n_w']] = DEFVAL
            data[n['chi2per']] = PDEFVAL
            data[n['aic']] = BIG_PDEFVAL
            data[n['bic']] = BIG_PDEFVAL

            data[n['tau']] = PDEFVAL

            if self.do_shear:
                data[n['g_sens']] = DEFVAL
                data[n['P']] = DEFVAL
                data[n['Q']] = DEFVAL
                data[n['R']] = DEFVAL

     
        self.data=data


    # methods below here not used
    def _extract_sub_lists(self,
                           keep_lol0,
                           im_lol0,
                           wt_lol0,
                           jacob_lol0):
        """
        extract those that passed some previous cuts
        """
        im_lol=[]
        wt_lol=[]
        jacob_lol=[]
        len_list=[]
        for band in self.iband:
            keep_list = keep_lol0[band]

            imlist0 = im_lol0[band]
            wtlist0 = wt_lol0[band]
            jacob_list0 = jacob_lol0[band]

            imlist = [imlist0[i] for i in keep_list]
            wtlist = [wtlist0[i] for i in keep_list]
            jacob_list = [jacob_list0[i] for i in keep_list]

            im_lol.append( imlist )
            wt_lol.append( wtlist )
            jacob_lol.append( jacob_list )

            len_list.append( len(imlist) )

        return im_lol, wt_lol, jacob_lol, len_list


    def _get_imlol_wtlol(self, dindex, mindex):
        """
        Get a list of the jacobians for this object
        skipping the coadd
        """

        im_lol=[]
        wt_lol=[]
        coadd_lol=[]
        coadd_wt_lol=[]

        for band in self.iband:
            meds=self.meds_list[band]

            # inherited functions
            imlist,coadd_imlist=self._get_imlist(meds,mindex)
            wtlist,coadd_wtlist=self._get_wtlist(meds,mindex)

            if self.reject_outliers:
                nreject=reject_outliers(imlist,wtlist)
                if nreject > 0:
                    print('        rejected:',nreject)

            self.data['nimage_tot'][dindex,band] = len(imlist)

            im_lol.append(imlist)
            wt_lol.append(wtlist)
            coadd_lol.append(coadd_imlist)
            coadd_wt_lol.append(coadd_wtlist)

        
        return im_lol,wt_lol,coadd_lol,coadd_wt_lol

    def _get_imlist(self, meds, mindex, type='image'):
        """
        get the image list, skipping the coadd
        """
        imlist_all = meds.get_cutout_list(mindex,type=type)

        coadd_imlist = [ imlist_all[0].astype('f8') ]
        se_imlist  = imlist_all[self.imstart:]

        se_imlist = [im.astype('f8') for im in se_imlist]
        return se_imlist, coadd_imlist


    def _get_wtlist(self, meds, mindex):
        """
        get the weight list.

        If using the seg map, mark pixels outside the coadd object region as
        zero weight
        """
        if self.region=='seg_and_sky':
            wtlist_all=meds.get_cweight_cutout_list(mindex)

            coadd_wtlist  = [ wtlist_all[0].astype('f8') ]
            se_wtlist     = wtlist_all[self.imstart:]

            se_wtlist=[wt.astype('f8') for wt in se_wtlist]
        else:
            raise ValueError("support other region types")
        return se_wtlist, coadd_wtlist

    def _get_jacobian_lol(self, mindex):
        """
        Get a list of the jacobians for this object
        skipping the coadd
        """

        jacob_lol=[]
        jacob_coadd_lol=[]
        for band in self.iband:
            meds=self.meds_list[band]

            jacob_list,coadd_jacob_list = self._get_jacobian_list(meds,mindex)
            jacob_lol.append(jacob_list)
            jacob_coadd_lol.append(coadd_jacob_list)

        return jacob_lol, jacob_coadd_lol

    def _get_jacobian_list(self, meds, mindex):
        """
        Get a list of the jacobians for this object
        skipping the coadd
        """
        jlist0=meds.get_jacobian_list(mindex)

        jlist_all=[]
        for jdict in jlist0:
            #print jdict
            j=ngmix.Jacobian(jdict['row0'],
                             jdict['col0'],
                             jdict['dudrow'],
                             jdict['dudcol'],
                             jdict['dvdrow'],
                             jdict['dvdcol'])
            jlist_all.append(j)

        jcoadd_jlist   = [jlist_all[0]]
        se_jlist       =  jlist_all[self.imstart:]
        return se_jlist, jcoadd_jlist

    def _fit_psfs(self, dindex, jacob_lol, do_coadd=False):
        """
        fit psfs for all bands
        """
        keep_lol=[]
        gmix_lol=[]
        flag_list=[]

        #self.numiter_sum=0.0
        #self.num=0
        for band in self.iband:
            meds=self.meds_list[band]
            jacob_list=jacob_lol[band]
            psfex_list=self.psfex_lol[band]

            keep_list, gmix_list, flags = self._fit_psfs_oneband(meds,
                                                                 dindex,
                                                                 band,
                                                                 jacob_list,
                                                                 psfex_list,
                                                                 do_coadd=do_coadd)


            keep_lol.append( keep_list )
            gmix_lol.append( gmix_list )

            # only propagate flags if we have no psfs left
            if len(keep_list) == 0:
                flag_list.append( flags )
            else:
                flag_list.append( 0 )

       
        return keep_lol, gmix_lol, flag_list


    def _fit_psfs_oneband(self,meds,dindex,band,jacob_list,psfex_list, do_coadd=False):
        """
        Generate psfex images for all SE images and fit
        them to gaussian mixture models

        We write the psf results into the psf structure *if*
        do_coadd==False
        """
        ptuple = self._get_psfex_reclist(meds, psfex_list, dindex,do_coadd=do_coadd)
        imlist,cenlist,siglist,flist,rng=ptuple

        keep_list=[]
        gmix_list=[]

        flags=0

        gmix_psf=None
        mindex = self.index_list[dindex]

        for i in xrange(len(imlist)):

            im=imlist[i]
            jacob0=jacob_list[i]
            sigma=siglist[i]
            icut=rng[i]

            cen0=cenlist[i]
            # the dimensions of the psfs are different, need
            # new center
            jacob=jacob0.copy()
            jacob._data['row0'] = cen0[0]
            jacob._data['col0'] = cen0[1]

            tflags=0
            try:
                fitter=self._do_fit_psf(im,jacob,sigma,first_guess=gmix_psf)

                gmix_psf=fitter.get_gmix()
                if not do_coadd:
                    self._set_psf_result(gmix_psf)

                keep,offset_arcsec=self._should_keep_psf(gmix_psf)
                if keep:
                    gmix_list.append( gmix_psf )
                    keep_list.append(i)
                else:
                    print( ('large psf offset: %s '
                                    'in %s' % (offset_arcsec,flist[i])) )
                    tflags |= PSF_LARGE_OFFSETS 

                
            except GMixMaxIterEM:
                print('psf fail',flist[i])

                tflags = PSF_FIT_FAILURE

            flags |= tflags

            if not do_coadd:
                self._set_psf_data(meds, mindex, band, icut, tflags)
                self.psf_index += 1

        return keep_list, gmix_list, flags


    def _get_psfex_reclist(self, meds, psfex_list, dindex, do_coadd=False):
        """
        Generate psfex reconstructions for the SE images
        associated with the cutouts
        """

        mindex = self.index_list[dindex]
        ncut=meds['ncutout'][mindex]
        imlist=[]
        cenlist=[]
        siglist=[]
        flist=[]

        if do_coadd:
            rng=[0]
        else:
            rng=range(1,ncut)

        for icut in rng:
            file_id=meds['file_id'][mindex,icut]
            pex=psfex_list[file_id]
            fname=pex['filename']

            row=meds['orig_row'][mindex,icut]
            col=meds['orig_col'][mindex,icut]

            im=pex.get_rec(row,col)
            cen=pex.get_center(row,col)

            imlist.append( im )
            cenlist.append(cen)
            siglist.append( pex.get_sigma() )
            flist.append( fname)

        return imlist, cenlist, siglist, flist, rng

    def _do_fit_psf(self, im, jacob, sigma_guess, first_guess=None):
        """
        old

        Fit a single psf
        """
        s2=sigma_guess**2
        im_with_sky, sky = ngmix.em.prep_image(im)

        fitter=ngmix.em.GMixEM(im_with_sky, jacobian=jacob)

        for i in xrange(self.psf_ntry):

            if i == 0 and first_guess is not None:
                gm_guess=first_guess.copy()
            else:
                s2guess=s2*jacob._data['det'][0]
                gm_guess=self._get_em_guess(s2guess)
            try:
                fitter.go(gm_guess, sky,
                          maxiter=self.psf_maxiter,
                          tol=self.psf_tol)
                break
            except GMixMaxIterEM:
                res=fitter.get_result()
                print('last fit:')
                print( fitter.get_gmix() )
                print( 'try:',i+1,'fdiff:',res['fdiff'],'numiter:',res['numiter'] )
                if i == (self.psf_ntry-1):
                    raise

        return fitter






_stat_names=['s2n_w',
             'chi2per',
             'dof',
             'aic',
             'bic']


_psf_ngauss_map={'em1':1, 'em2':2}
def get_psf_ngauss(psf_model):
    if psf_model not in _psf_ngauss_map:
        raise ValueError("bad psf model: '%s'" % psf_model)
    return _psf_ngauss_map[psf_model]


def _get_as_list(data_in):
    if not isinstance(data_in, list):
        out=[data_in]
    else:
        out=data_in
    return out

def calc_offset_arcsec(gm, scale=1.0):
    data=gm.get_data()

    offset=numpy.sqrt( (data['row'][0]-data['row'][1])**2 + 
                       (data['col'][0]-data['col'][1])**2 )
    offset_arcsec=offset*scale
    return offset_arcsec


_em2_fguess=numpy.array([0.5793612389470884,1.621860687127999])
_em2_pguess=numpy.array([0.596510042804182,0.4034898268889178])
_em3_pguess = numpy.array([0.596510042804182,0.4034898268889178,1.303069003078001e-07])
_em3_fguess = numpy.array([0.5793612389470884,1.621860687127999,7.019347162356363],dtype='f8')



#_em2_fguess=numpy.array([12.6,3.8])
#_em2_fguess[:] /= _em2_fguess.sum()
#_em2_pguess=numpy.array([0.30, 0.70])


def reject_outliers(imlist, wtlist, nsigma=5.0, A=0.3):
    """
    Set the weight for outlier pixels to zero

     | im - med | > n*sigma_i + A*|med|

    where mu is the median

    I actually do

        wt*(im-med)**2 > (n + A*|med|*sqrt(wt))**2

    We wrongly assume the images all align, but this is ok as long as nsigma is
    high enough

    If the number of images is < 3 then the weight maps are not modified
    """

    nreject=0

    nim=len(imlist)
    if nim < 3:
        return nreject

    dims=imlist[0].shape
    imstack = numpy.zeros( (nim, dims[0], dims[1]) )

    for i,im in enumerate(imlist):
        imstack[i,:,:] = im

    med=numpy.median(imstack, axis=0)

    for i in xrange(nim):
        im=imlist[i]
        wt=wtlist[i]

        wt.clip(0.0, out=wt)

        ierr = numpy.sqrt(wt)

        # wt*(im-med)**2
        chi2_image = im.copy()
        chi2_image -= med
        chi2_image *= chi2_image
        chi2_image *= wt

        # ( n + A*|med|*sqrt(wt) )**2
        maxvals = numpy.abs(med)
        maxvals *= A
        maxvals *= ierr
        maxvals += nsigma
        maxvals *= maxvals

        w=numpy.where(chi2_image > maxvals)

        if w[0].size > 0:
            wt[w] = 0.0
            nreject += w[0].size

    return nreject

