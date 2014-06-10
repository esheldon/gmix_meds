"""
todo

    - better to guess from coadd or em1?
    - allow shear expand?  Only works for constant shear

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
from ngmix import GMixMaxIterEM, GMixRangeError, print_pars
from ngmix import Observation, ObsList, MultiBandObsList

from .lmfit import get_model_names

# starting new values for these
DEFVAL=-9999
PDEFVAL=9999
BIG_DEFVAL=-9.999e9
BIG_PDEFVAL=9.999e9


NO_CUTOUTS=2**0
PSF_FIT_FAILURE=2**1
PSF_LARGE_OFFSETS=2**2
EXP_FIT_FAILURE=2**3
DEV_FIT_FAILURE=2**4

BOX_SIZE_TOO_BIG=2**5

EM_FIT_FAILURE=2**6

LOW_PSF_FLUX=2**7

UTTER_FAILURE=2**8

NO_ATTEMPT=2**30

#PSF_S2N=1.e6
PSF_OFFSET_MAX=0.25
PSF_TOL=1.0e-5
EM_MAX_TRY=3
EM_MAX_ITER=100

_CHECKPOINTS_DEFAULT_MINUTES=[30,60,110]

class UtterFailure(Exception):
    """
    could not make a good guess
    """
    def __init__(self, value):
         self.value = value
    def __str__(self):
        return repr(self.value)


class MedsFit(dict):
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

        self.meds_files=_get_as_list(meds_files)

        self['nband']=len(self.meds_files)
        self.iband = range(self['nband'])

        self._unpack_priors(priors)

        self._load_meds_files()

        self.obj_range=obj_range
        self._set_index_list()

        self.psfex_lol = self._get_psfex_lol()

        self._set_some_defaults()

        self.checkpoint_file=checkpoint_file
        self.checkpoint_data=checkpoint_data
        self._setup_checkpoints()

        self.random_state=numpy.random.RandomState()

        if self.checkpoint_data is None:
            self._make_struct()
            self._make_epoch_struct()

    def _set_some_defaults(self):
        self['psf_offset_max']=self.get('psf_offset_max',PSF_OFFSET_MAX)
        self['region']=self.get('region','cweight-nearest')
        self['max_box_size']=self.get('max_box_size',2048)
        self['reject_outliers']=self.get('reject_outliers',True) # from cutouts
        self['make_plots']=self.get('make_plots',False)

    def _unpack_priors(self, priors_in):
        """
        Currently only separable priors
        """

        from ngmix.joint_prior import PriorSimpleSep
        from ngmix.priors import ZDisk2D

        priors={}
        gflat_priors={}

        if 'joint_prior' in priors_in:
            print("NEED TO SUPPORT MULTIPLE BANDS FOR JOINT")
            prior=priors_in['joint_prior']
            gflat_prior=priors_in['joint_prior_gflat']

            for i,model in enumerate(self['fit_models']):
                priors[model]=prior
                gflat_priors[model]=gflat_prior

                # extra for the coadd gaussian fit
                if i==1:
                    priors["gauss"] = prior
                    gflat_priors["gauss"] = gflat_prior


        else:
            cen_prior=priors_in['cen_prior']

            counts_prior_repeat=self.get('counts_prior_repeat',False)

            g_prior_flat=ZDisk2D(1.0)

            g_priors=priors_in['g_priors']
            T_priors=priors_in['T_priors']
            counts_priors=priors_in['counts_priors']

            models = self['fit_models']
            nmod=len(models)

            nprior=len(g_priors)
            if nprior != nmod:
                raise ValueError("len(models)=%d but got len(priors)=%d" % (nmod,nprior))

            for i in xrange(nmod):
                model=self['fit_models'][i]
                print("loading prior for:",model)

                cp = counts_priors[i]
                if counts_prior_repeat:
                    cp = [cp]*self['nband']

                print("    full")
                prior = PriorSimpleSep(cen_prior,
                                       g_priors[i],
                                       T_priors[i],
                                       cp)

                # for the exploration, for which we do not apply g prior during
                print("    gflat")
                gflat_prior = PriorSimpleSep(cen_prior,
                                             g_prior_flat,
                                             T_priors[i],
                                             cp)

                priors[model]=prior
                gflat_priors[model]=gflat_prior

                # for the coadd gaussian fit
                if i==0:
                    priors["gauss"] = prior
                    gflat_priors["gauss"] = gflat_prior

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

        if len(coadd_mb_obs_list) != self['nband']:
            print("  some coadd failed to fit psf")
            self.data['flags'][dindex] = PSF_FIT_FAILURE 
            return

        if len(mb_obs_list) == 0:
            print("  not all bands had at least one psf fit succeed")
            self.data['flags'][dindex] = PSF_FIT_FAILURE 
            return

        print(coadd_mb_obs_list[0][0].image.shape)

        self.data['nimage_use'][dindex, :] = n_im

        self.sdata={'coadd_mb_obs_list':coadd_mb_obs_list,
                    'mb_obs_list':mb_obs_list}
        self.dindex=dindex

        try:
            flags=self._fit_all_models()
        except UtterFailure as err:
            print("Got utter failure error: %s" % str(err))
            flags=UTTER_FAILURE

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
        if box_size > self['max_box_size']:
            print('Box size too big:',box_size)
            flags |= BOX_SIZE_TOO_BIG

        if meds['ncutout'][mindex] < 1:
            print('No cutouts')
            flags |= NO_CUTOUTS
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
                if self['reject_outliers']:
                    self._reject_outliers(obs_list)
                mb_obs_list.append(obs_list)
            n_im += this_n_im

        return coadd_mb_obs_list, mb_obs_list, n_im

    def _reject_outliers(self, obs_list):
        imlist=[]
        wtlist=[]
        for obs in obs_list:
            imlist.append(obs.image)
            wtlist.append(obs.weight)

        # weight map is modified
        nreject=meds.reject_outliers(imlist,wtlist)
        if nreject > 0:
            print('        rejected:',nreject)


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

        # for the psf fitting code
        wt.clip(min=0.0, max=None, out=wt)

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

        empars=self['psf_em_pars']
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
        if self['region']=='seg_and_sky':
            wt=meds.get_cweight_cutout(mindex, icut)
        elif self['region']=="cweight-nearest":
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
        psf_ngauss=self['psf_em_pars']['ngauss']
        if psf_ngauss == 2:
            offset_arcsec = calc_offset_arcsec(gm)
            if offset_arcsec > self['psf_offset_max']:
                keep=False

        return keep, offset_arcsec


    def _fit_all_models(self):
        """
        Fit psf flux and other models
        """

        flags=0
        # fit both coadd and se psf flux if exists
        self._fit_psf_flux()

        dindex=self.dindex
        s2n=self.data['coadd_psf_flux'][dindex,:]/self.data['coadd_psf_flux_err'][dindex,:]
        max_s2n=numpy.nanmax(s2n)

        n_se_images=len(self.sdata['mb_obs_list'])
         
        if max_s2n >= self['min_psf_s2n']:
            # we use this as a guess for the real galaxy models
            print("    fitting coadd gauss")
            self._run_model_fit('gauss', coadd=True)

            for model in self['fit_models']:
                print('    fitting:',model)

                print('    coadd')
                self._run_model_fit(model, coadd=True)

                if self['fit_me_galaxy'] and n_se_images > 0:
                    print('    multi-epoch')
                    self._run_model_fit(model, coadd=False)
        else:
            mess="    psf s/n too low: %s (%s)"
            mess=mess % (max_s2n,self['min_psf_s2n'])
            print(mess)
            
            flags |= LOW_PSF_FLUX

        return flags


    '''
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
        empars=self['galaxy_em_pars']

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

    '''

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
            except GMixRangeError as e:
                print("            em: range: %s" % str(e))
                if i == (ntry-1):
                    raise GMixMaxIterEM("too many iter")

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




    def _fit_psf_flux(self):
        """
        Perform PSF flux fits on each band separately
        """

        sdata=self.sdata
        dindex=self.dindex

        print('    fitting: psf')
        for band,obs_list in enumerate(sdata['coadd_mb_obs_list']):
            self._fit_psf_flux_oneband(dindex, band, obs_list,coadd=True)

        # this can be zero length
        for band,obs_list in enumerate(sdata['mb_obs_list']):
            self._fit_psf_flux_oneband(dindex, band, obs_list,coadd=False)


    def _fit_psf_flux_oneband(self, dindex, band, obs_list, coadd=False):
        """
        Fit the PSF flux in a single band
        """
        if coadd:
            name='coadd_psf'
        else:
            name='psf'

        fitter=ngmix.fitting.TemplateFluxFitter(obs_list, do_psf=True)
        fitter.go()

        res=fitter.get_result()
        data=self.data

        n=get_model_names(name)
        data[n['flags']][dindex,band] = res['flags']
        data[n['flux']][dindex,band] = res['flux']
        data[n['flux_err']][dindex,band] = res['flux_err']
        data[n['chi2per']][dindex,band] = res['chi2per']
        data[n['dof']][dindex,band] = res['dof']
        print("        %s flux(%s): %g +/- %g" % (name,band,res['flux'],res['flux_err']))



    def _run_model_fit(self, model, coadd=False):
        """
        wrapper to run fit, copy pars, maybe make plots

        sets .coadd_gauss_fitter or .fitter or .coadd_fitter
        """
        if coadd:
            if model=='gauss':
                self.guesser=self._get_guesser('coadd_psf')
            else:
                self.guesser=self._get_guesser('coadd_gauss')
            mb_obs_list=self.sdata['coadd_mb_obs_list']
        else:
            self.guesser=self._get_guesser(self['guess_type'])
            mb_obs_list=self.sdata['mb_obs_list']

        fitter=self._fit_model(mb_obs_list, model)

        self._copy_simple_pars(fitter, coadd=coadd)

        self._print_res(fitter, coadd=coadd)

        if self['make_plots']:
            self._do_make_plots(fitter, model, coadd=coadd)

        if coadd:
            if model=='gauss':
                self.coadd_gauss_fitter=fitter
            else:
                self.coadd_fitter=fitter
        else:
            self.fitter=fitter

    def _fit_model(self, mb_obs_list, model):
        """
        Fit all the simple models
        """


        fitter=self._fit_simple_emcee(mb_obs_list, model)

        # also adds .weights attribute
        self._calc_mcmc_stats(fitter, model)

        log_res=fitter.get_result()
        lin_res=fitter.get_lin_result()

        if self['do_shear']:
            self._add_shear_info(log_res, fitter, model)

        return fitter

    def _get_guesser(self, guess_type):
        if guess_type=='coadd_psf':
            guesser=self._get_guesser_from_coadd_psf()

        elif guess_type=='coadd_gauss':
            guesser=self._get_guesser_from_coadd_gauss()
        elif guess_type=='coadd_gauss_best':
            guesser=self._get_guesser_from_coadd_gauss_best()


        elif guess_type=='coadd_mcmc':
            guesser=self._get_guesser_from_coadd_mcmc()
        elif guess_type=='coadd_mcmc_best':
            guesser=self._get_guesser_from_coadd_mcmc_best() 
        elif guess_type=='coadd_lm':
            guesser=self._get_guesser_from_coadd_lm()
        else:
            raise ValueError("bad guess type: '%s'" % guess_type)

        return guesser

    def _get_guesser_from_priors(self):
        """
        the guesser just draws random values from the priors
        """

        return self.prior.sample

    def _get_guesser_from_coadd_psf(self):
        """
        take flux guesses from psf take canonical center (0,0)
        and near zero ellipticity.  Size is taken from around the
        expected psf size, which is about 0.9''

        The size will often be too big

        """
        print('        getting guess from coadd psf')

        dindex=self.dindex
        data=self.data

        psf_flux=data['coadd_psf_flux'][dindex,:].copy()
        psf_flux=psf_flux.clip(min=0.1, max=1.0e6)

        nband=self['nband']
        w,=numpy.where(data['coadd_psf_flags'][dindex,:] != 0)
        if w.size > 0:
            print("    found %s/%s psf failures" % (w.size,nband))
            if w.size == psf_flux.size:
                val=5.0
                print("setting all to default:",val)
            else:
                wgood,=numpy.where(data['coadd_psf_flags'][dindex,:] == 0)
                val=numpy.median(psf_flux[wgood])
                print("setting to median:",val)

            psf_flux[w] = val

        # arbitrary
        T = 2*(0.9/2.35)**2

        guesser=FromPSFGuesser(T, psf_flux)
        return guesser

    def _get_guesser_from_coadd_mcmc(self):
        """
        get a random set of points from the coadd chain
        """

        print('        getting guess from coadd mcmc')

        fitter=self.coadd_fitter
        log_trials = fitter.get_trials()

        res=fitter.get_result()
        sigmas=res['pars_err']

        guesser=FromMCMCGuesser(log_trials, sigmas)
        return guesser

    def _get_guesser_from_coadd_mcmc_best(self):
        """
        get a random set of points from the coadd chain
        """

        print('        getting guess from coadd mcmc best')

        fitter=self.coadd_fitter
        best_pars=fitter.best_pars

        res=fitter.get_result()
        sigmas=res['pars_err']

        guesser=FromParsGuesser(best_pars, sigmas)
        return guesser


    def _get_guesser_from_coadd_gauss(self):
        """
        get a random set of points from the coadd chain
        """

        print('        getting guess from coadd gauss')

        fitter=self.coadd_gauss_fitter
        log_trials = fitter.get_trials()

        res=fitter.get_result()
        sigmas=res['pars_err']

        guesser=FromMCMCGuesser(log_trials, sigmas)
        return guesser

    def _get_guesser_from_coadd_gauss_best(self):
        """
        get a random set of points from the coadd chain
        """

        print('        getting guess from coadd gauss best')

        fitter=self.coadd_gauss_fitter
        best_pars=fitter.best_pars

        res=fitter.get_result()
        sigmas=res['pars_err']

        guesser=FromParsGuesser(best_pars, sigmas)
        return guesser


    def _get_guesser_from_coadd_lm(self):
        """
        get a random set of points from the coadd chain
        """

        print('        getting guess from coadd lm')

        fitter=self.coadd_fitter_lm

        # in log space
        res=fitter.get_result()

        if res['flags'] != 0:
            raise RuntimeError("    whoops, lm failed but you asked for a lm guess!")
            #print("        LM failed, falling back to psf for guess")
            #guesser=self._get_guesser_from_coadd_psf()
        else:
            pars=res['pars']
            sigmas=res['pars_err']

            guesser=FromParsGuesser(res['pars'], res['pars_err'])
        return guesser

    def _fit_simple_emcee(self, mb_obs_list, model):
        """
        Fit one of the "simple" models, e.g. exp or dev

        use flat g prior
        """

        from ngmix.fitting import MCMCSimple

        # note flat on g!
        prior=self.gflat_priors[model]

        guess=self.guesser(n=self['emcee_nwalkers'], prior=prior)

        fitter=MCMCSimple(mb_obs_list,
                          model,
                          prior=prior,
                          nwalkers=self['emcee_nwalkers'],
                          mca_a=self['emcee_a'],
                          random_state=self.random_state)

        pos=fitter.run_mcmc(guess,self['emcee_burnin'])
        pos=fitter.run_mcmc(pos,self['emcee_nstep'])

        return fitter


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



    def _copy_simple_pars(self, fitter, coadd=False):
        """
        Copy from the result dict to the output array
        """

        dindex=self.dindex
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
            lin_pars=lin_res['pars']
            lin_pars_cov=lin_res['pars_cov']

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

            for sn in _stat_names:
                self.data[n[sn]][dindex] = log_res[sn]

            # this stuff won't be in the result for LM fitting
            if 'arate' in log_res:
                self.data[n['arate']][dindex] = log_res['arate']
                if log_res['tau'] is not None:
                    self.data[n['tau']][dindex] = log_res['tau']

                if self['do_shear']:
                    self.data[n['g_sens']][dindex,:] = log_res['g_sens']
                    self.data[n['P']][dindex] = log_res['P']
                    self.data[n['Q']][dindex,:] = log_res['Q']
                    self.data[n['R']][dindex,:,:] = log_res['R']
     

               

    def _do_make_plots(self, fitter, model, coadd=False,
                       fitter_type='emcee'):
        """
        make plots
        """

        if fitter_type in ['emcee','mh']:
            do_trials=True
        else:
            do_trials=False

        dindex=self.dindex
        if coadd:
            type='coadd'
        else:
            type='mb'

        type = '%s-%s' % (type,fitter_type)

        mindex = self.index_list[dindex]

        title='%s %s' % (type,model)
        res_plots=fitter.plot_residuals(title=title)

        if do_trials:
            pdict=fitter.make_plots(title=title,
                                    weights=fitter.weights)


            trials_png='%s-trials-%06d-%s.png' % (type,mindex,model)
            wtrials_png='%s-wtrials-%06d-%s.png' % (type,mindex,model)

            print("            ",trials_png)
            pdict['trials'].write_img(1200,1200,trials_png)

            print("            ",wtrials_png)
            pdict['wtrials'].write_img(1200,1200,wtrials_png)


        if res_plots is not None:
            for band, band_plots in enumerate(res_plots):
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


    def _print_res(self, fitter, coadd=False):
        lin_res=fitter.get_lin_result()
        if lin_res['flags']==0:
            log_res=fitter.get_result()
            if coadd:
                type='coadd'
            else:
                type='mb'

            print("        %s logpars:" % type)
            print_pars(log_res['pars'],    front='        ')
            print_pars(log_res['pars_err'],front='        ')

            print("        %s linear pars:" % type)
            print_pars(lin_res['pars'],    front='        ')
            print_pars(lin_res['pars_err'],front='        ')
            if 'arate' in lin_res:
                print('        arate:',lin_res['arate'])

    def _setup_checkpoints(self):
        """
        Set up the checkpoint times in minutes and data

        self.checkpoint_data and self.checkpoint_file
        """
        self.checkpoints = self.get('checkpoints',_CHECKPOINTS_DEFAULT_MINUTES)
        self.n_checkpoint    = len(self.checkpoints)
        self.checkpointed    = [0]*self.n_checkpoint

        self._set_checkpoint_data()

        if self.checkpoint_file is not None:
            self.do_checkpoint=True
        else:
            self.do_checkpoint=False

    def _set_checkpoint_data(self):
        """
        See if checkpoint data was sent

        self._checkpoint_data should be set on construction
        """
        if self.checkpoint_data is not None:
            self.data=self.checkpoint_data['data']

            # for nband==1 the written array drops the arrayness
            self.data.dtype=self._get_dtype()
            self.epoch_data=self.checkpoint_data['epoch_data']

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
        for model in self['fit_models']:
            if model not in ['exp','dev']:
                raise ValueError("model '%s' not supported" % model)

    def _get_all_models(self):
        """
        get all model names, includeing the coadd_ ones
        """
        models=['coadd_%s' % model for model in self['fit_models']]

        if self['fit_me_galaxy']:
            models = models + self['fit_models']

        models = ['coadd_gauss'] + models

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

        psf_ngauss=self['psf_em_pars']['ngauss']
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

        nband=self['nband']
        bshape=(nband,)
        simple_npars=5+nband

        dt=[('number','i4'),
            ('processed','i1'),
            ('flags','i4'),
            ('nimage_tot','i4',bshape),
            ('nimage_use','i4',bshape),
            ('time','f8')]

        # coadd fit with em 1 gauss
        # the psf flux fits are done for each band separately
        for name in ['coadd_psf','psf']:
            n=get_model_names(name)
            dt += [(n['flags'],   'i4',bshape),
                   (n['flux'],    'f8',bshape),
                   (n['flux_err'],'f8',bshape),
                   (n['chi2per'],'f8',bshape),
                   (n['dof'],'f8',bshape)]

        if nband==1:
            cov_shape=(nband,)
        else:
            cov_shape=(nband,nband)

        models=self._get_all_models()
        for model in models:

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
            if self['do_shear']:
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

        for name in ['coadd_psf','psf']:
            n=get_model_names(name)
            data[n['flags']] = NO_ATTEMPT
            data[n['flux']] = DEFVAL
            data[n['flux_err']] = PDEFVAL
            data[n['chi2per']] = PDEFVAL

        models=self._get_all_models()
        for model in models:
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

            if self['do_shear']:
                data[n['g_sens']] = DEFVAL
                data[n['P']] = DEFVAL
                data[n['Q']] = DEFVAL
                data[n['R']] = DEFVAL

     
        self.data=data


    #
    #
    # methods below here not used
    #
    #

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

            if self['reject_outliers']:
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
        se_imlist  = imlist_all[self['imstart']:]

        se_imlist = [im.astype('f8') for im in se_imlist]
        return se_imlist, coadd_imlist


    def _get_wtlist(self, meds, mindex):
        """
        get the weight list.

        If using the seg map, mark pixels outside the coadd object region as
        zero weight
        """
        if self['region']=='seg_and_sky':
            wtlist_all=meds.get_cweight_cutout_list(mindex)

            coadd_wtlist  = [ wtlist_all[0].astype('f8') ]
            se_wtlist     = wtlist_all[self['imstart']:]

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
        se_jlist       =  jlist_all[self['imstart']:]
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



class MHMedsFitLM(MedsFit):
    """
    This version uses MH for fitting, with guesses from a maxlike fit
    """
    def __init__(self,
                 conf,
                 priors,
                 meds_files,
                 **keys):
        super(MHMedsFit,self).__init__(conf,priors,meds_files,**keys)
        self['emcee_nwalkers']=1

    def _get_all_models(self):
        """
        get all model names, includeing the coadd_ ones
        """
        models=['coadd_%s' % model for model in self['fit_models']]

        if self['fit_me_galaxy']:
            models = models + self['fit_models']

        return models


    def _fit_all_models(self):
        """
        Fit psf flux and other models
        """

        flags=0
        # fit both coadd and se psf flux if exists
        self._fit_psf_flux()

        dindex=self.dindex
        s2n=self.data['coadd_psf_flux'][dindex,:]/self.data['coadd_psf_flux_err'][dindex,:]
        max_s2n=numpy.nanmax(s2n)

        n_se_images=len(self.sdata['mb_obs_list'])
         
        if max_s2n >= self['min_psf_s2n']:
            for model in self['fit_models']:
                print('    fitting:',model)

                print('    coadd lm')
                self._run_model_fit(model, coadd=True, fitter_type='lm')
                res=self.coadd_fitter_lm.get_result()
                if res['flags']==0:
                    print('    coadd mcmc')
                    self._run_model_fit(model, coadd=True, fitter_type='mh')

                    if self['fit_me_galaxy'] and n_se_images > 0:
                        print('    multi-epoch lm')
                        self._run_model_fit(model, coadd=False, fitter_type='lm')

                        res=self.fitter_lm.get_result()
                        if res['flags']==0:
                            print('    multi-epoch mcmc')
                            self._run_model_fit(model, coadd=False, fitter_type='mh')
                        else:
                            print("    LM ME failed, skipping mcmc fit")

                else:
                    print("    LM coadd failed, skipping mcmc and ME fits")
        else:
            mess="    psf s/n too low: %s (%s)"
            mess=mess % (max_s2n,self['min_psf_s2n'])
            print(mess)
            
            flags |= LOW_PSF_FLUX

        return flags

    def _run_model_fit(self, model, coadd=False, fitter_type='mh'):
        """
        wrapper to run fit, copy pars, maybe make plots

        sets .coadd_gauss_fitter or .fitter or .coadd_fitter

        this one does not currently use self['guess_type']
        """

        if coadd:
            if fitter_type=='lm':
                self.guesser=self._get_guesser('coadd_psf')
                # this might be better for total crap objects
                #self.guesser=self.priors[model].sample
            else:
                self.guesser=self._get_guesser('coadd_lm')
            mb_obs_list=self.sdata['coadd_mb_obs_list']
        else:
            self.guesser=self._get_guesser('coadd_mcmc')
            mb_obs_list=self.sdata['mb_obs_list']

        fitter=self._fit_model(mb_obs_list,
                               model,
                               fitter_type=fitter_type)

        if fitter_type=='mh':
            self._copy_simple_pars(fitter, coadd=coadd)
        else:
            print("        NOT COPYING PARS BECAUSE LM: make lin pars...")

        self._print_res(fitter, coadd=coadd)

        if self['make_plots']:
            self._do_make_plots(fitter, model, coadd=coadd,
                                fitter_type=fitter_type)

        if coadd:
            if model=='gauss':
                self.coadd_gauss_fitter=fitter
            else:
                if fitter_type=='lm':
                    self.coadd_fitter_lm=fitter
                else:
                    self.coadd_fitter=fitter
        else:
            if fitter_type=='lm':
                self.fitter_lm=fitter
            else:
                self.fitter=fitter

    def _fit_model(self, mb_obs_list, model, fitter_type='mh'):
        """
        Fit all the simple models
        """

        if fitter_type=='mh':
            fitter=self._fit_simple_emcee(mb_obs_list, model)
        elif fitter_type=='lm':
            fitter=self._fit_simple_lm(mb_obs_list, model)
        else:
            raise ValueError("bad fitter type: '%s'" % fitter_type)

        # also adds .weights attribute
        if fitter_type=='mh':
            self._calc_mcmc_stats(fitter, model)
            lin_res=fitter.get_lin_result()

            log_res=fitter.get_result()

            if self['do_shear']:
                self._add_shear_info(log_res, fitter, model)

        return fitter


    def _fit_simple_emcee(self, mb_obs_list, model):
        """
        Fit one of the "simple" models, e.g. exp or dev

        use flat g prior
        """

        from ngmix.fitting import MHSimple

        # note flat on g!
        prior=self.gflat_priors[model]

        guess,sigmas=self.guesser(get_sigmas=True)

        step_sizes = 0.5*sigmas

        max_step = 0.5*self.priors[model].get_widths()
        print_pars(max_step, front="        max_step:")

        for i in xrange(guess.size):
            step_sizes[i] = step_sizes[i].clip(max=max_step[i])

        print_pars(step_sizes, front="        step sizes:")

        fitter=MHSimple(mb_obs_list,
                        model,
                        step_sizes,
                        prior=prior,
                        random_state=self.random_state)

        pos=fitter.run_mcmc(guess,self['mh_burnin'])
        pos=fitter.run_mcmc(pos,self['mh_nstep'])

        return fitter

    def _fit_simple_lm(self, mb_obs_list, model):
        """
        Fit one of the "simple" models, e.g. exp or dev

        use flat g prior
        """

        from ngmix.fitting import LMSimple

        # note the prior is *not* flat in g! This is important to constrain the
        # covariance matrix

        prior=self.priors[model]

        ntry=self['gal_lm_ntry']
        for i in xrange(ntry):
            guess=self.guesser()
            print_pars(guess, front='            lm guess:')

            fitter=LMSimple(mb_obs_list,
                            model,

                            prior=prior,

                            lm_pars=self['gal_lm_pars'])

            fitter.run_lm(guess)
            res=fitter.get_result()
            if res['flags']==0:
                break

        res['ntry']=i+1
        return fitter

class MHMedsFitHybrid(MedsFit):
    """
    This version uses MH for fitting, with guess/steps from
    a single-gauss emcee run
    """

    def _fit_all_models(self):
        """
        Fit psf flux and other models
        """

        flags=0
        # fit both coadd and se psf flux if exists
        self._fit_psf_flux()

        dindex=self.dindex
        s2n=self.data['coadd_psf_flux'][dindex,:]/self.data['coadd_psf_flux_err'][dindex,:]
        max_s2n=numpy.nanmax(s2n)

        n_se_images=len(self.sdata['mb_obs_list'])
         
        if max_s2n >= self['min_psf_s2n']:

            # we use this as a guess for the real galaxy models, and for
            # step sizes
            print("    fitting coadd gauss")
            self._run_model_fit('gauss', coadd=True, fitter_type='emcee')

            for model in self['fit_models']:
                print('    fitting:',model)

                print('    coadd')
                self._run_model_fit(model, coadd=True, fitter_type='mh')

                if self['fit_me_galaxy'] and n_se_images > 0:
                    print('    multi-epoch')
                    self._run_model_fit(model, coadd=False, fitter_type='mh')

        else:
            mess="    psf s/n too low: %s (%s)"
            mess=mess % (max_s2n,self['min_psf_s2n'])
            print(mess)
            
            flags |= LOW_PSF_FLUX

        return flags

    def _run_model_fit(self, model, coadd=False, fitter_type='mh'):
        """
        wrapper to run fit, copy pars, maybe make plots

        sets .coadd_gauss_fitter or .fitter or .coadd_fitter

        this one does not currently use self['guess_type']
        """

        if coadd:
            if fitter_type=='emcee' and model=='gauss':
                # we need to bootstrap
                self.guesser=self._get_guesser('coadd_psf')
            else:
                # get guess and step sizes from coadd gauss fit from
                # emcee
                self.guesser=self._get_guesser('coadd_gauss_best')
            mb_obs_list=self.sdata['coadd_mb_obs_list']
        else:
            # take our guess from the coadd mh fit
            self.guesser=self._get_guesser('coadd_mcmc_best')
            mb_obs_list=self.sdata['mb_obs_list']

        fitter=self._fit_model(mb_obs_list,
                               model,
                               fitter_type=fitter_type)

        self._copy_simple_pars(fitter, coadd=coadd)

        self._print_res(fitter, coadd=coadd)

        if self['make_plots']:
            self._do_make_plots(fitter, model, coadd=coadd, fitter_type=fitter_type)

        if coadd:
            if model=='gauss':
                self.coadd_gauss_fitter=fitter
            else:
                self.coadd_fitter=fitter
        else:
            self.fitter=fitter

    def _fit_model(self, mb_obs_list, model, fitter_type='mh'):
        """
        Fit all the simple models
        """

        if fitter_type=='emcee':
            fitter=self._fit_simple_emcee(mb_obs_list, model)
        elif fitter_type=='mh':
            fitter=self._fit_simple_mh(mb_obs_list, model)
        else:
            raise ValueError("bad fitter type: '%s'" % fitter_type)

        self._calc_mcmc_stats(fitter, model)
        lin_res=fitter.get_lin_result()

        log_res=fitter.get_result()

        if self['do_shear']:
            self._add_shear_info(log_res, fitter, model)

        return fitter


    def _fit_simple_mh(self, mb_obs_list, model):
        """
        Fit one of the "simple" models, e.g. exp or dev

        use flat g prior
        """

        from ngmix.fitting import MHSimple

        # note flat on g!
        prior=self.gflat_priors[model]

        guess,sigmas=self.guesser(get_sigmas=True, prior=prior)

        step_sizes = 0.5*sigmas

        max_step = 0.5*self.priors[model].get_widths()
        print_pars(max_step, front="        max_step:")

        for i in xrange(guess.size):
            step_sizes[i] = step_sizes[i].clip(max=max_step[i])

        print_pars(step_sizes, front="        step sizes:")

        fitter=MHSimple(mb_obs_list,
                        model,
                        step_sizes,
                        prior=prior,
                        random_state=self.random_state)

        pos=fitter.run_mcmc(guess,self['mh_burnin'])
        pos=fitter.run_mcmc(pos,self['mh_nstep'])

        return fitter


class GuesserBase(object):
    def _fix_guess(self, guess, prior, ntry=4):
        #guess[:,2]=-9999
        n=guess.shape[0]
        for j in xrange(n):
            for itry in xrange(ntry):

                try:
                    lnp=prior.get_lnprob_scalar(guess[j,:])
                except GMixRangeError as err:
                    print_pars(guess[j,:], front="bad guess:")
                    if itry < ntry:
                        guess[j,:] = prior.sample()
                    else:
                        raise UtterFailure("could not find a good guess")


class FromMCMCGuesser(GuesserBase):
    """
    get guesses from a set of trials
    """
    def __init__(self, trials, sigmas):
        self.trials=trials
        self.sigmas=sigmas

    def __call__(self, n=1, get_sigmas=False, prior=None):
        """
        get a random set of points from the trials
        """
        import random

        trials=self.trials
        np = trials.shape[0]

        rand_int = random.sample(xrange(np), n)
        guess=trials[rand_int, :]

        if prior is not None:
            self._fix_guess(guess, prior)

        if get_sigmas:
            return guess, self.sigmas
        else:
            return guess

class FromPSFGuesser(GuesserBase):
    """
    get full guesses from just T,fluxes associated with
    psf

    should make this take log values...
    """
    def __init__(self, T, fluxes):
        self.T=T
        self.fluxes=fluxes

        self.log_T = numpy.log10(T)
        self.log_fluxes = numpy.log10(fluxes)

    def __call__(self, n=1, prior=None, **keys):
        """
        center, shape are just distributed around zero
        """
        fluxes=self.fluxes
        nband=fluxes.size
        np = 5+nband

        guess=numpy.zeros( (n, np) )
        guess[:,0] = 0.01*srandu(n)
        guess[:,1] = 0.01*srandu(n)
        guess[:,2] = 0.1*srandu(n)
        guess[:,3] = 0.1*srandu(n)
        #guess[:,4] = numpy.log10( self.T*(1.0 + 0.2*srandu(n)) )
        guess[:,4] = self.log_T + 0.1*srandu(n)

        for band in xrange(nband):
            #guess[:,5+band] = numpy.log10( fluxes[band]*(1.0 + 0.1*srandu(n)) )
            guess[:,5+band] = self.log_fluxes[band] + 0.1*srandu(n)

        if prior is not None:
            self._fix_guess(guess, prior)

        if n==1:
            guess=guess[0,:]
        return guess

class FromParsGuesser(GuesserBase):
    """
    get full guesses from just T,fluxes associated with
    psf
    """
    def __init__(self, pars, pars_err):
        self.pars=pars
        self.pars_err=pars_err

    def __call__(self, n=None, get_sigmas=False, prior=None):
        """
        center, shape are just distributed around zero
        """

        if n is None:
            n=1
            is_scalar=True
        else:
            is_scalar=False

        pars=self.pars
        npars=pars.size

        width = pars*0 + 0.01

        guess=numpy.zeros( (n, npars) )

        guess[:,0] = width[0]*srandu(n)
        guess[:,1] = width[1]*srandu(n)

        guess_shape=get_shape_guess(pars[2],pars[3],n,width[2:2+2])
        guess[:,2]=guess_shape[:,0]
        guess[:,3]=guess_shape[:,1]

        # we add to log pars!
        for i in xrange(4,npars):
            guess[:,i] = pars[i] + width[i]*srandu(n)

        if prior is not None:
            self._fix_guess(guess, prior)

        if is_scalar:
            guess=guess[0,:]

        if get_sigmas:
            return guess, self.pars_err
        else:
            return guess


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

def get_shape_guess(g1, g2, n, width):
    """
    Get guess, making sure in range
    """

    guess=numpy.zeros( (n, 2) )
    shape=ngmix.Shape(g1, g2)

    for i in xrange(n):

        while True:
            try:
                g1_offset = width[0]*srandu()
                g2_offset = width[1]*srandu()
                shape_new=shape.copy()
                shape_new.shear(g1_offset, g2_offset)
                break
            except GMixRangeError:
                pass

        guess[i,0] = shape_new.g1
        guess[i,1] = shape_new.g2

    return guess


