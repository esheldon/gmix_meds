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
from ngmix import GMixModel, GMix

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

IMAGE_FLAGS=2**9

NO_ATTEMPT=2**30

#PSF_S2N=1.e6
PSF_OFFSET_MAX=0.25
PSF_TOL=1.0e-5
EM_MAX_TRY=3
EM_MAX_ITER=100

# shift psf flags past astrometry flags, which end at 9
PSFEX_FLAGS_SHIFT = 9

_CHECKPOINTS_DEFAULT_MINUTES=[0,30,60,110]

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
                 model_data=None,
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

        self.meds_files=get_as_list(meds_files)

        self['nband']=len(self.meds_files)
        self.iband = range(self['nband'])

        self._unpack_priors(priors)

        self.obj_range=obj_range
        self.model_data=model_data
        self.checkpoint_file=checkpoint_file
        self.checkpoint_data=checkpoint_data

        self._set_some_defaults()

        # load meds files and image flags array
        self._load_meds_files()
        self._maybe_load_coadd_cat_files()

        self._set_index_list()

        self.psfex_lists, self.psfex_flags_lists = self._get_psfex_lists()

        self._combine_image_flags()

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
        if self['reject_outliers']:
            print("will reject outliers")
        self['make_plots']=self.get('make_plots',False)

        self['work_dir'] = self.get('work_dir',os.environ.get('TMPDIR','/tmp'))

        self['check_image_flags']=self.get('check_image_flags',False)

        self['use_psf_rerun']=self.get('use_psf_rerun',False)

        if self.model_data is not None:
            self['model_neighbors']=True
        else:
            self['model_neighbors']=False

    def _reset_mb_sums(self):
        from numpy import zeros
        nband=self['nband']

        self.coadd_npix               = 0.0
        self.coadd_wsum               = 0.0
        self.coadd_wrelsum            = 0.0
        #self.coadd_psfrec_counts_wsum = zeros(nband,dtype='f8')
        self.coadd_psfrec_T_wsum      = 0.0
        self.coadd_psfrec_g1_wsum     = 0.0
        self.coadd_psfrec_g2_wsum     = 0.0

        self.npix               = 0.0
        self.wsum               = 0.0
        self.wrelsum            = 0.0
        self.psfrec_T_wsum      = 0.0
        self.psfrec_g1_wsum     = 0.0
        self.psfrec_g2_wsum     = 0.0

    def _unpack_priors(self, priors_in):
        """
        Currently only separable priors
        """

        from ngmix.joint_prior import PriorSimpleSep
        from ngmix.priors import ZDisk2D

        priors={}
        gflat_priors={}

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

        self.dindex=dindex
        self.mindex = self.index_list[dindex]

        # for checkpointing
        self.data['processed'][dindex]=1

        mindex = self.index_list[dindex]

        ncutout_tot=self._get_object_ncutout(mindex)
        self.data['nimage_tot'][dindex, :] = ncutout_tot

        # need to do this because we work on subset files
        self.data['id'][dindex] = self.meds_list[0]['id'][mindex]
        self.data['number'][dindex] = self.meds_list[0]['number'][mindex]
        self.data['box_size'][dindex] = \
                self.meds_list[0]['box_size'][mindex]
        print('coadd_objects_id: %ld' % self.data['id'][dindex])

        flags = self._obj_check(mindex)
        if flags != 0:
            self.data['flags'][dindex] = flags
            return 0
        
        #need coadd seg maps here to model nbrs
        if self['model_neighbors']:
            self._set_coadd_seg_maps()
        
        # MultiBandObsList obects
        coadd_mb_obs_list, mb_obs_list, n_im = \
                self._get_multi_band_observations(mindex)

        if len(coadd_mb_obs_list) != self['nband']:
            print("  some coadd failed to fit psf")
            self.data['flags'][dindex] = PSF_FIT_FAILURE 
            return

        if len(mb_obs_list) != self['nband']:
            print("  not all bands had at least one psf fit"
                  " succeed and were without image flags")
            self.data['flags'][dindex] = PSF_FIT_FAILURE 
            return

        print(coadd_mb_obs_list[0][0].image.shape)

        self.data['nimage_use'][dindex, :] = n_im[:]

        self.sdata={'coadd_mb_obs_list':coadd_mb_obs_list,
                    'mb_obs_list':mb_obs_list}

        try:
            flags=self._fit_all_models()
        except UtterFailure as err:
            print("Got utter failure error: %s" % str(err))
            flags=UTTER_FAILURE

        self.data['flags'][dindex] = flags
        self.data['time'][dindex] = time.time()-t0

    def _set_coadd_seg_maps(self):
        mindex=self.mindex
        self._coadd_seg_maps=[]
        for m in self.meds_list:
            seg=m.get_cutout(mindex, 0, type='seg')
            self._coadd_seg_maps.append(seg)

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

        if max_s2n >= self['min_psf_s2n'] and len(self['fit_models']) > 0:
            for model in self['fit_models']:
                print('    fitting:',model)

                print('    coadd')
                self._run_model_fit(model, coadd=True)

                if self['fit_me_galaxy']:
                    print('    multi-epoch')
                    self._run_model_fit(model, coadd=False)
        else:
            mess="    psf s/n too low: %s (%s)"
            mess=mess % (max_s2n,self['min_psf_s2n'])
            print(mess)
            
            flags |= LOW_PSF_FLUX

        return flags

    def _run_model_fit(self, model, coadd=False):
        """
        wrapper to run fit, copy pars, maybe make plots

        sets .fitter or .coadd_fitter
        """
        if coadd:
            self.guesser=self._get_guesser(self['coadd_model_guess'])
            mb_obs_list=self.sdata['coadd_mb_obs_list']
        else:
            self.guesser=self._get_guesser(self['me_model_guess'])
            mb_obs_list=self.sdata['mb_obs_list']

        fitter=self._fit_model(mb_obs_list, model)

        self._copy_simple_pars(fitter, coadd=coadd)

        self._print_res(fitter, coadd=coadd)

        if self['make_plots']:
            self._do_make_plots(fitter, model, coadd=coadd)

        if coadd:
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

        if self['do_shear']:
            self._add_shear_info(fitter, model)

        return fitter

    def _fit_simple_emcee(self, mb_obs_list, model):
        """
        Fit one of the "simple" models, e.g. exp or dev

        use flat g prior
        """

        from ngmix.fitting import MCMCSimple

        # note flat on g!
        prior=self.gflat_priors[model]

        epars=self['emcee_pars']
        guess=self.guesser(n=epars['nwalkers'], prior=prior)
        fmt = "%.6f "*(5+self['nband'])
        print("        emcee guess: ",fmt%tuple(numpy.mean(guess,axis=0)))
        #for olist in mb_obs_list:
        #    print("    image filename:",olist[0].filename)
        #    print("    psfex filename:",olist[0].psf.filename)

        fitter=MCMCSimple(mb_obs_list,
                          model,
                          nu=self['nu'],
                          prior=prior,
                          nwalkers=epars['nwalkers'],
                          mca_a=epars['a'],
                          random_state=self.random_state)

        pos=fitter.run_mcmc(guess,epars['burnin'])
        pos=fitter.run_mcmc(pos,epars['nstep'])

        p = fitter.get_best_pars()
        print("        emcee final: ",fmt%tuple(p))
        
        return fitter


    def _get_object_ncutout(self, mindex):
        """
        number of cutouts for the specified object.
        """
        ncutout=numpy.zeros(self['nband'],dtype='i4')
        for i,meds in enumerate(self.meds_list):
            ncutout[i] = meds['ncutout'][mindex]
        return ncutout

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

        box_size=meds['box_size'][mindex]
        if box_size > self['max_box_size']:
            print('Box size too big:',box_size)
            flags |= BOX_SIZE_TOO_BIG

        # need coadd and at least one SE image
        ncutout=meds['ncutout'][mindex]
        if ncutout < 2:
            print('No cutouts')
            flags |= NO_CUTOUTS

        # note coadd is never flagged
        image_flags=self._get_image_flags(band, mindex)
        w,=numpy.where(image_flags==0)

        # need coadd and at lease one SE image
        if w.size < 2:
            print('< 2 with no image flags')
            flags |= IMAGE_FLAGS

        # informative only
        if w.size != image_flags.size:
            print("    for band %d removed %d/%d images due to "
                  "flags" % (band, ncutout-w.size, ncutout))

        return flags



    def _get_multi_band_observations(self, mindex):
        """
        Get an ObsList object for the Coadd observations
        Get a MultiBandObsList object for the SE observations.
        """

        dindex=self.dindex
        coadd_mb_obs_list=MultiBandObsList()
        mb_obs_list=MultiBandObsList()

        # number used in each band
        n_im = numpy.zeros(self['nband'],dtype='i4')

        self._reset_mb_sums()

        # only append if good ones found, can use to demand the length is
        # nband.  But we want to finish to set the psfrec info
        for band in self.iband:

            #self.coadd_band_wsum=0.0
            #self.coadd_psfrec_counts_wsum=0.0

            #self.band_wsum=0.0

            cobs_list, obs_list = self._get_band_observations(band, mindex)

            if len(cobs_list) > 0:
                coadd_mb_obs_list.append(cobs_list)

            nme = len(obs_list)
            n_im[band] = nme

            if nme > 0:
                if self['reject_outliers']:
                    self._reject_outliers(obs_list)
                mb_obs_list.append(obs_list)

        # means must go accross bands
        self.set_psf_means()
        
        print("        mask_frac:",self.data['mask_frac'][dindex],
              "coadd mask_frac:",self.data['coadd_mask_frac'][dindex])

        if self['model_neighbors']:
            print("    modelling neighbors:")
            print("        doing coadd:")
            self._model_neighbors(coadd_mb_obs_list, coadd=True)
            print("        doing SE:")
            self._model_neighbors(mb_obs_list)

        return coadd_mb_obs_list, mb_obs_list, n_im

    def _model_neighbors(self, mb_obs_list, coadd=False):
        """
        model the neighbors

        need the full object_data from the meds file, as well as a results
        structure holding the fits
        """
        
        mindex_local = self.mindex #index in current meds file
        
        #stuff to get names
        nexp = Namer('exp')
        ndev = Namer('dev')
        if self['nbrs_model']['model'] == 'exp':
            nmodel = nexp
            model = 'exp'
        elif self['nbrs_model']['model'] == 'dev':
            nmodel = ndev
            model = 'dev'

        ncoadd = Namer('coadd')
        nme = Namer('')
        if coadd:
            ntot = ncoadd
        else:
            ntot = nme

        # import code here
        for band, obs_list in enumerate(mb_obs_list):
            print("            doing band %d" % band)
            seg = self._coadd_seg_maps[band]
            mod = self.model_data['meds_object_data'][band]
            meds = self.meds_list[band]
            number = meds['number'][mindex_local] #number for seg map, index+1 into entire meds file
            mindex_global = number-1
            
            # get objects that are in this object's segmentation map.
            # this will return nothing if there were no neighbors in the
            # postage stamp
            w=numpy.where((seg > 0) & (seg != number))
            if w[0].size > 0:
                
                w=numpy.where(seg > 0)
                ids=numpy.unique(seg[w]) - 1
                for obs in obs_list:
                    #copy old image and weight map
                    obs.image_orig = obs.image.copy()
                    obs.weight_orig = obs.weight.copy()
                    icut_cen = obs.meta['icut']
                    fid_cen = meds['file_id'][mindex_local,icut_cen]
                    tot_image = numpy.zeros(obs.image.shape)
                    cen_image = None
                    
                    for cid in ids:
                        #check all flags first
                        if self.model_data['model_fits'][self['nbrs_model']['flags']][cid] == 0:
                            
                            #if have extra info, check its flags
                            if 'model_extra_info' in self.model_data:
                                if self.model_data['model_extra_info']['flags'][cid] != 0:
                                    continue
                            
                            #logic for best_chi2per
                            #if both flags != 0; skip
                            # otherwise pick model with zero flags
                            # othrwise pick best
                            if self['nbrs_model']['model'] == 'best_chi2per':
                                if self.model_data['model_fits'][ntot(nexp(self['nbrs_model']['flags']))][cid] != 0 \
                                        and self.model_data['model_fits'][ntot(ndev(self['nbrs_model']['flags']))][cid] != 0:
                                    continue
                                elif self.model_data['model_fits'][ntot(nexp(self['nbrs_model']['flags']))][cid] != 0:
                                    nmodel = ndev
                                    model = 'dev'
                                elif self.model_data['model_fits'][ntot(ndev(self['nbrs_model']['flags']))][cid] != 0:
                                    nmodel = nexp
                                    model = 'exp'
                                elif self.model_data['model_fits'][ntot(nexp('chi2per'))][cid] > \
                                        self.model_data['model_fits'][ntot(ndev('chi2per'))][cid]:
                                    nmodel = ndev
                                    model = 'dev'
                                else:
                                    nmodel = nexp
                                    model = 'exp'
                            
                            #always reject models with bad flags
                            if self.model_data['model_fits'][ntot(nmodel(self['nbrs_model']['flags']))][cid] != 0:
                                continue
                            
                            #see if need good ME fit 
                            if 'require_me_goodfit' in self['nbrs_model']:
                                if self['nbrs_model']['require_me_goodfit']:
                                    if self.model_data['model_fits'][nme(nmodel(self['nbrs_model']['flags']))][cid] != 0:
                                        continue
                            
                            ##################################################
                            #render each object in the seg map with a good fit
                            
                            #get cutout with same file_id as central object
                            icut_obj, = numpy.where(mod['file_id'][cid] == fid_cen)
                            if len(icut_obj) == 0:
                                print("                could not find cutout for nbr %d file_id %d" % (cid,fid_cen))
                                continue
                            if len(icut_obj) > 1:
                                print("                found duplicate cutouts for nbr %d file_id %d" % (cid,fid_cen))
                                assert len(icut_obj) == 1, "found duplicate cutouts for nbr %d file_id %d!" % (cid,fid_cen)
                            icut_obj = icut_obj[0]
                            
                            #find psf entry and make sure it is OK 
                            q, = numpy.where((self.model_data['epochs']['number'] == cid+1) & 
                                             (self.model_data['epochs']['file_id'] == fid_cen) & 
                                             (self.model_data['epochs']['band_num'] == band))
                            #skip if psf fit was bad or could not find psf
                            if len(q) == 0:
                                print("                could not find PSF fit for nbr %d cutout %d" % (cid,icut_obj))
                                continue                            
                            if len(q) > 1:
                                print("                found duplicate PSF fits for nbr %d cutout %d" % (cid,icut_obj))
                                assert len(q) == 1, "found duplicate PSF fits for nbr %d cutout %d" % (cid,icut_obj)
                            if self.model_data['epochs']['psf_fit_flags'][q[0]] > 0:
                                continue
                            pars_psf = self.model_data['epochs']['psf_fit_pars'][q[0]]
                            
                            #fiducial location of object in postage stamp
                            row = mod['orig_row'][cid,icut_obj] - obs.meta['orig_start_row']
                            col = mod['orig_col'][cid,icut_obj] - obs.meta['orig_start_col']
                            
                            #parameters for object
                            pars_obj = self.model_data['model_fits'][ntot(nmodel(self['nbrs_model']['pars']))][cid]                            
                            pinds = range(5)
                            pinds.append(band+5)
                            pars_obj = pars_obj[pinds] 
                            
                            #get jacobian
                            jacob = Jacobian(row,col,
                                             mod['dudrow'][cid,icut_obj],
                                             mod['dudcol'][cid,icut_obj],
                                             mod['dvdrow'][cid,icut_obj],
                                             mod['dvdcol'][cid,icut_obj])
                            pixscale = jacob.get_scale()
                            row += pars_obj[0]/pixscale
                            col += pars_obj[1]/pixscale
                            jacob.set_cen(row,col)
                            
                            #now render image of object
                            psf_gmix = GMix(pars=pars_psf)
                            gmix_sky = GMixModel(pars_obj, model)
                            gmix_image = gmix_sky.convolve(psf_gmix)
                            obj_image = gmix_image.make_image(obs.image.shape, jacobian=jacob)
                            
                            tot_image += obj_image
                            if cid == mindex_global:
                                cen_image = obj_image.copy()
                            
                            if self['nbrs_model']['method'] == 'subtract':
                                #subtract its flux if not central
                                if cid != mindex_global:
                                    obs.image -= obj_image

                            
                    if self['make_plots']:
                        def plot_seg(seg):
                            seg_new = seg.copy()
                            seg_new = seg_new.astype(float)
                            uvals = numpy.unique(seg)
                            mval = 1.0*(len(uvals)-1)
                            ind = 1.0
                            for uval in uvals:
                                if uval > 0:
                                    qx,qy = numpy.where(seg == uval)
                                    seg_new[qx[:],qy[:]] = ind/mval
                                    ind += 1
                                    
                            return seg_new

                        import images
                        import biggles
                        width = 1920
                        height = 1200
                        biggles.configure('screen','width', width)
                        biggles.configure('screen','height', height)
                        tab = biggles.Table(2,3)
                        tab.title = 'coadd_objects_id = %d' % mod['id'][mindex_global]
                        
                        if cen_image is None:
                            cen_image = numpy.zeros_like(obs.image)
                        
                        tab[0,0] = images.view(obs.image_orig,title='original image',show=False)
                        tab[0,1] = images.view(tot_image-cen_image,title='models of nbrs',show=False)
                        if coadd:
                            tab[0,2] = images.view(plot_seg(seg),title='seg map',show=False)
                        else:
                            tab[0,2] = images.view(plot_seg(meds.interpolate_coadd_seg(mindex_local,icut_cen)),title='seg map',show=False)
                        
                        tab[1,0] = images.view(obs.image,title='corrected image',show=False)
                        msk = tot_image != 0
                        frac = numpy.zeros(tot_image.shape)
                        frac[msk] = cen_image[msk]/tot_image[msk]
                        tab[1,1] = images.view(frac,title='fraction of flux due to central',show=False)
                        tab[1,2] = images.view(obs.weight,title='weight map',show=False)
                        
                        if icut_cen > 0:
                            tab.write_img(1920,1200,'%06d-nbrs-model-band%d-icut%d.png' % (mindex_global,band,icut_cen))
                            print("                %06d-nbrs-model-band%d-icut%d.png" % (mindex_global,band,icut_cen))
                        else:
                            tab.write_img(1920,1200,'%06d-nbrs-model-band%d-coadd.png' % (mindex_global,band))
                            print("                %06d-nbrs-model-band%d-coadd.png" % (mindex_global,band))

                        if False:
                            tab.show()
                            import ipdb
                            ipdb.set_trace()

    def set_psf_means(self):
        dindex=self.dindex

        # if npix == 0 there was some problem, and mask_frac would
        # not be calculable

        npix=self.coadd_npix
        wsum=self.coadd_wsum
        if npix > 0:
            mask_frac = 1.0 - self.coadd_wrelsum/npix
        else:
            mask_frac=PDEFVAL

        if wsum > 0:
            iwsum = 1.0/wsum

            T  = self.coadd_psfrec_T_wsum*iwsum
            g1 = self.coadd_psfrec_g1_wsum*iwsum
            g2 = self.coadd_psfrec_g2_wsum*iwsum

        else:
            wsum=0.0
            T=DEFVAL
            g1=DEFVAL
            g2=DEFVAL

        self.data['coadd_mask_frac'][dindex]=mask_frac
        self.data['coadd_psfrec_T'][dindex]=T
        self.data['coadd_psfrec_g'][dindex,0]=g1
        self.data['coadd_psfrec_g'][dindex,1]=g2

        npix=self.npix
        wsum=self.wsum

        if npix > 0:
            mask_frac = 1.0 - self.wrelsum/npix
        else:
            mask_frac=PDEFVAL

        if wsum > 0:
            iwsum = 1.0/wsum

            T  = self.psfrec_T_wsum*iwsum
            g1 = self.psfrec_g1_wsum*iwsum
            g2 = self.psfrec_g2_wsum*iwsum

        else:
            wsum=0.0
            T=DEFVAL
            g1=DEFVAL
            g2=DEFVAL

        self.data['mask_frac'][dindex]=mask_frac
        self.data['psfrec_T'][dindex]=T
        self.data['psfrec_g'][dindex,0]=g1
        self.data['psfrec_g'][dindex,1]=g2


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

        note we have already checked that we have a coadd and a single epoch
        without flags
        """

        meds=self.meds_list[band]
        ncutout=meds['ncutout'][mindex]

        image_flags=self._get_image_flags(band, mindex)

        coadd_obs_list = ObsList()
        obs_list       = ObsList()

        for icut in xrange(ncutout):
            iflags = image_flags[icut]
            if iflags != 0:
                flags = IMAGE_FLAGS
            else:
                try:
                    obs = self._get_band_observation(band, mindex, icut)

                    if icut==0:
                        coadd_obs_list.append( obs )
                    else:
                        obs_list.append(obs)
                    flags=0
                except GMixMaxIterEM:
                    flags=PSF_FIT_FAILURE

            # we set the metadata even if the fit fails
            self._set_psf_meta(meds, mindex, band, icut, flags)
            self.epoch_index += 1


        '''
        icut=0
        iflags=image_flags[icut]
        if iflags != 0:
            flags = IMAGE_FLAGS
        else:
            try:
                coadd_obs = self._get_band_observation(band, mindex, icut)
                coadd_obs_list.append( coadd_obs )
                flags=0
            except GMixMaxIterEM:
                flags |= PSF_FIT_FAILURE

        self._set_psf_meta(meds, mindex, band, icut, flags)
        self.epoch_index += 1

        for icut in xrange(1,ncutout):
            iflags = image_flags[icut]
            if iflags != 0:
                flags = IMAGE_FLAGS
            else:
                try:
                    obs = self._get_band_observation(band, mindex, icut)
                    obs_list.append(obs)
                    flags=0
                except GMixMaxIterEM:
                    flags=PSF_FIT_FAILURE

            # we set the metadata even if the fit fails
            self._set_psf_meta(meds, mindex, band, icut, flags)
            self.epoch_index += 1
        '''

        return coadd_obs_list, obs_list

    def _get_band_observation(self, band, mindex, icut):
        """
        Get an Observation for a single band.

        GMixMaxIterEM is raised if psf fitting fails
        """
        meds=self.meds_list[band]

        fname = self._get_meds_orig_filename(meds, mindex, icut)
        im = self._get_meds_image(meds, mindex, icut)
        wt = self._get_meds_weight(meds, mindex, icut)
        jacob = self._get_jacobian(meds, mindex, icut)

        # for the psf fitting code
        wt=wt.clip(min=0.0)

        psf_obs = self._get_psf_observation(band, mindex, icut, jacob)

        psf_fitter = self._fit_psf(psf_obs)
        psf_gmix = psf_fitter.get_gmix()

        # we only get here if psf fitting succeeds because an exception gets
        # raised on fit failure; note other metadata should always get set
        # above.  relies on global variable self.epoch_index
        #
        # note this means that the psf counts sum and wsum should always
        # be incremented together

        npix = im.size

        wsum = wt.sum()
        wmax = wt.max()
        imsum = psf_obs.image.sum()

        g1,g2,T=psf_gmix.get_g1g2T()

        if icut==0:
            self.coadd_npix += npix
            #self.coadd_psfrec_counts_wsum[band] += imsum*wsum
            self.coadd_psfrec_T_wsum += T*wsum
            self.coadd_psfrec_g1_wsum += g1*wsum
            self.coadd_psfrec_g2_wsum += g2*wsum
            self.coadd_wsum += wsum
            #self.coadd_wsum_byband[band] += wsum

            #if wmax > self.coadd_wmax_byband[band]:
            #    self.coadd_wmax_byband[band]=wmax
            if wmax > 0.0:
                self.coadd_wrelsum += wsum/wmax
        else:
            self.npix += npix
            self.psfrec_T_wsum += T*wsum
            self.psfrec_g1_wsum += g1*wsum
            self.psfrec_g2_wsum += g2*wsum
            self.wsum += wsum
            #self.wsum_byband[band] += wsum

            #if wmax > self.wmax_byband[band]:
            #    self.wmax_byband[band]=wmax
            if wmax > 0.0:
                self.wrelsum += wsum/wmax

        self._set_psf_result(psf_gmix, imsum)
        self._set_wsum_wmax_npix(wsum,wmax,npix)

        psf_obs.set_gmix(psf_gmix)

        obs=Observation(im,
                        weight=wt,
                        jacobian=jacob,
                        psf=psf_obs)

        obs.filename=fname

        meta={'icut':icut,
              'orig_start_row':meds['orig_start_row'][mindex, icut],
              'orig_start_col':meds['orig_start_col'][mindex, icut]}
        obs.update_meta_data(meta)

        psf_fwhm=2.35*numpy.sqrt(psf_gmix.get_T()/2.0)
        print("        psf fwhm:",psf_fwhm)

        if self['make_plots']:
            self._do_make_psf_plots(band, psf_gmix, psf_obs, mindex, icut)

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
        im, cen, sigma_pix, fname = self._get_psf_image(band, mindex, icut)

        psf_jacobian = image_jacobian.copy()
        psf_jacobian.set_cen(cen[0], cen[1])

        psf_obs = Observation(im, jacobian=psf_jacobian)
        psf_obs.filename=fname

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

        pex=self.psfex_lists[band][file_id]
        #print("    using psfex from:",pex['filename'])

        row=meds['orig_row'][mindex,icut]
        col=meds['orig_col'][mindex,icut]

        im=pex.get_rec(row,col).astype('f8', copy=False)
        cen=pex.get_center(row,col)
        sigma_pix=pex.get_sigma()

        return im, cen, sigma_pix, pex['filename']

    def _get_meds_orig_filename(self, meds, mindex, icut):
        """
        Get the original filename
        """
        file_id=meds['file_id'][mindex, icut]
        ii=meds.get_image_info()
        return ii['image_path'][file_id]

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
        elif self['region']=='weight':
            wt=meds.get_cutout(mindex, icut, type='weight')
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



    def _set_psf_result(self, gm, counts):
        """
        Set psf fit data.

        im is the psf image
        """

        epoch_index=self.epoch_index

        pars=gm.get_full_pars()
        g1,g2,T=gm.get_g1g2T()

        ed=self.epoch_data

        ed['psf_counts'][epoch_index]    = counts
        #print("        psf counts:",ed['psf_counts'][epoch_index])
        ed['psf_fit_g'][epoch_index,0]    = g1
        ed['psf_fit_g'][epoch_index,1]    = g2
        ed['psf_fit_T'][epoch_index]      = T
        ed['psf_fit_pars'][epoch_index,:] = pars


    def _set_wsum_wmax_npix(self, wsum, wmax, npix):
        """
        set weight sum and max for this epoch
        """

        epoch_index=self.epoch_index
        self.epoch_data['npix'][epoch_index] = npix
        self.epoch_data['wsum'][epoch_index] = wsum
        self.epoch_data['wmax'][epoch_index] = wmax

        #print("        wsum:",wsum)


    def _set_psf_meta(self, meds, mindex, band, icut, flags):
        """
        Set all the meta data for the psf result
        """
        epoch_index=self.epoch_index
        ed=self.epoch_data

        # mindex can be an index into a sub-range meds
        ed['id'][epoch_index] = meds['id'][mindex]
        ed['number'][epoch_index] = meds['number'][mindex]
        ed['band_num'][epoch_index] = band
        ed['cutout_index'][epoch_index] = icut
        ed['orig_row'][epoch_index] = meds['orig_row'][mindex,icut]
        ed['orig_col'][epoch_index] = meds['orig_col'][mindex,icut]
        ed['psf_fit_flags'][epoch_index] = flags

        file_id  = meds['file_id'][mindex,icut].astype('i4')
        image_id = meds._image_info[file_id]['image_id']
        ed['file_id'][epoch_index]  = file_id
        ed['image_id'][epoch_index]  = image_id


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




    def _get_guesser(self, guess_type):
        if guess_type=='coadd_psf':
            guesser=self._get_guesser_from_coadd_psf()

        elif guess_type=='coadd_cat':
            guesser=self._get_guesser_from_coadd_cat()

        elif guess_type=='coadd_mcmc':
            guesser=self._get_guesser_from_coadd_mcmc()

        elif guess_type=='coadd_mcmc_best':
            guesser=self._get_guesser_from_coadd_mcmc_best() 

        #elif guess_type=='coadd_lm':
        #    guesser=self._get_guesser_from_coadd_lm()

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
        psf_flux=psf_flux.clip(min=0.1, max=1.0e9)

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

    def _get_coadd_cat_best(self):
        """
        get T guess based on flux radius in the guess band
        """
        from .constants import PIXSCALE, PIXSCALE2
        dindex=self.dindex
        mindex = self.index_list[dindex]

        c=self._cat_list[self['T_guess_band']]
        flux_radius = c['flux_radius'][mindex]

        # in arcsec
        sigma = 2.0*flux_radius/2.35*PIXSCALE

        T = 2*sigma**2

        # fluxes also need to be converted
        fluxes = [c['flux_model'][mindex] for c in self._cat_list]
        fluxes = PIXSCALE2*numpy.array(fluxes, dtype='f8')
        return T, fluxes


    def _get_guesser_from_coadd_cat(self):
        """
        Take flux guesses from coadd catalogs and size from the guess_band flux_radius
        """
        print('        getting guess from coadd catalog')

        dindex=self.dindex
        mindex = self.index_list[dindex]

        T, fluxes = self._get_coadd_cat_best()
        print("        coadd T:",T,"coadd fluxes:",fluxes)

        guesser=FromPSFGuesser(T, fluxes)
        return guesser


    def _get_guesser_from_coadd_mcmc(self):
        """
        get a random set of points from the coadd chain
        """

        print('        getting guess from coadd mcmc')

        fitter=self.coadd_fitter

        # trials in default scaling
        trials = fitter.get_trials()
        #lnprobs = fitter.get_lnprobs()

        # result with default scaling
        res=fitter.get_result()
        sigmas=res['pars_err']

        guesser=FromMCMCGuesser(trials, sigmas)
        return guesser

    def _get_guesser_from_coadd_mcmc_best(self):
        """
        guess based on best result from mcmc run
        """

        print('        getting guess from coadd mcmc best')

        fitter=self.coadd_fitter
        best_pars=fitter.get_best_pars()

        # keep the T and flux guesses from going too negative
        ncheck=best_pars.size - 4
        for i in xrange(ncheck):
            if best_pars[4+i] < 0.0:
                best_pars[4+i] = 0.001*srandu()

        res=fitter.get_result()
        sigmas=res['pars_err']

        guesser=FixedParsGuesser(best_pars, sigmas)
        return guesser

    '''
    def _get_guesser_from_coadd_lm(self):
        """
        get a random set of points from the coadd chain
        """

        print('        getting guess from coadd lm')

        fitter=self.coadd_fitter_lm

        # currently lm only works in in log space
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
    '''



    def _calc_mcmc_stats(self, fitter, model):
        """
        Add some stats for the mcmc chain

        Also add a weights attribute to the fitter
        """

        #do unweighted version first
        fitter.calc_result() 
        uw_result = fitter.get_result()
        fitter._unweighted_result = uw_result
        
        # trials in default scaling, should not matter
        trials=fitter.get_trials()

        g_prior=self.priors[model].g_prior
        weights = g_prior.get_prob_array2d(trials[:,2], trials[:,3])
        fitter.calc_result(weights=weights)

        fitter.weights=weights

    def _add_shear_info(self, fitter, model):
        """
        Add pqr or lensfit info
        """

        # result in default scaling
        res=fitter.get_result()

        # trials in default scaling, should not matter
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

            pars_best=fitter.get_best_pars()

            flux=pars[5:]
            flux_cov=pars_cov[5:, 5:]

            self.data[n('pars')][dindex,:] = pars
            self.data[n('pars_cov')][dindex,:,:] = pars_cov

            self.data[n('pars_best')][dindex,:] = pars_best


            self.data[n('flux')][dindex] = flux
            self.data[n('flux_cov')][dindex] = flux_cov

            self.data[n('g')][dindex,:] = res['g']
            self.data[n('g_cov')][dindex,:,:] = res['g_cov']

            for sn in _stat_names:
                self.data[n(sn)][dindex] = res[sn]

            # this stuff won't be in the result for LM fitting
            if 'arate' in res:
                self.data[n('arate')][dindex] = res['arate']
                if res['tau'] is not None:
                    self.data[n('tau')][dindex] = res['tau']

                if self['do_shear']:
                    self.data[n('g_sens')][dindex,:] = res['g_sens']
                    self.data[n('P')][dindex] = res['P']
                    self.data[n('Q')][dindex,:] = res['Q']
                    self.data[n('R')][dindex,:,:] = res['R']
        
        #copy in unweighted pars
        if hasattr(fitter, '_unweighted_result'):
            res=fitter._unweighted_result
            self.data[n('flags_uw')][dindex] = res['flags']
            
            if res['flags'] == 0:
                pars=res['pars']
                pars_cov=res['pars_cov']
                self.data[n('pars_uw')][dindex,:] = pars                
                self.data[n('pars_cov_uw')][dindex,:,:] = pars_cov
                self.data[n('chi2per_uw')][dindex] = res['chi2per']
                self.data[n('dof_uw')][dindex] = res['dof']
                if 'arate' in res:
                    self.data[n('arate_uw')][dindex] = res['arate']
                    if res['tau'] is not None:
                        self.data[n('tau_uw')][dindex] = res['tau']


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
        try:
            res_plots=fitter.plot_residuals(title=title)
            if res_plots is not None:
                for band, band_plots in enumerate(res_plots):
                    for icut, plt in enumerate(band_plots):
                        fname='%06d-%s-resid-%s-band%d-im%d.png' % (mindex,type,model,band,icut+1)
                        print("            ",fname)
                        plt.write_img(1920,1200,fname)

        except GMixRangeError as err:
            print("caught error plotting resid: %s" % str(err))

        if do_trials:
            try:
                pdict=fitter.make_plots(title=title,
                                        weights=fitter.weights)


                trials_png='%06d-%s-trials-%s.png' % (mindex,type,model)
                wtrials_png='%06d-%s-wtrials-%s.png' % (mindex,type,model)

                print("            ",trials_png)
                pdict['trials'].write_img(1200,1200,trials_png)

                print("            ",wtrials_png)
                pdict['wtrials'].write_img(1200,1200,wtrials_png)
            except:
                print("caught error plotting trials")



    def _do_make_psf_plots(self, band, gmix, obs, mindex, icut):
        """
        make residual plots for psf
        """
        import images

        if icut==0:
            type='coadd-psf'
        else:
            type='psf'

        title='%06d band: %s' % (mindex, band)
        if icut==0:
            title='%s coadd' % title
        else:
            title='%s %d' % (title,icut)

        im=obs.image

        model_im=gmix.make_image(im.shape, jacobian=obs.jacobian)
        modflux=model_im.sum()
        if modflux <= 0:
            print("psf model flux too low:",modflux)
            return

        model_im *= ( im.sum()/modflux )

        plt=images.compare_images(im, model_im,
                                  label1='psf', label2='model',
                                  show=False)
        plt.title=title

        if icut==0:
            fname='%06d-psf-resid-band%d-coadd.png' % (mindex,band)
        else:
            fname='%06d-psf-resid-band%d-icut%d.png' % (mindex,band,icut+1)

        print("            ",fname)
        plt.write_img(1920,1200,fname)


    def _combine_image_flags(self):
        """
        add the psf flags, properly shifted
        """
        for band in self.iband:
            image_flags = self.all_image_flags[band]
            psf_flags   = self.psfex_flags_lists[band]
            for i in xrange(image_flags.size):
                image_flags[i] |= psf_flags[i]

    def _get_image_flags(self, band, mindex):
        """
        find images associated with the object and get the image flags

        Also add in the psfex flags, eventually incorporated into meds
        """
        meds=self.meds_list[band]
        ncutout=meds['ncutout'][mindex]

        if self['check_image_flags']:
            file_ids = meds['file_id'][mindex, 0:ncutout]
            image_flags = self.all_image_flags[band][file_ids]
        else:
            image_flags = numpy.zeros(ncutout, dtype='i4')

        return image_flags

    def _load_meds_files(self):
        """
        Load all listed meds files
        """

        self.meds_list=[]
        self.meds_meta_list=[]
        self.all_image_flags=[]

        for i,f in enumerate(self.meds_files):
            print(f)
            medsi=meds.MEDS(f)
            medsi_meta=medsi.get_meta()
            image_info=medsi.get_image_info()

            if i==0:
                nobj_tot=medsi.size
            else:
                nobj=medsi.size
                if nobj != nobj_tot:
                    raise ValueError("mismatch in meds "
                                     "sizes: %d/%d" % (nobj_tot,nobj))
            self.meds_list.append(medsi)
            self.meds_meta_list.append(medsi_meta)
            self.all_image_flags.append( image_info['image_flags'].copy() )

        self.nobj_tot = self.meds_list[0].size

    def _maybe_load_coadd_cat_files(self):
        """
        load the catalogs for fit guesses
        """
        import fitsio
        if ('cat' in self['coadd_model_guess']
                or 'cat' in self['me_model_guess']):
            cat_list=[]
            for m in self.meds_list:
                image_info=m.get_image_info()
                image_path=image_info['image_path'][0].strip()
                cat_path=get_coadd_cat_path(image_path)

                print("loading catalog:",cat_path)
                cat=fitsio.read(cat_path, lower=True)

                cat_list.append(cat)

            self._cat_list=cat_list

    def _get_psfex_lists(self):
        """
        Load psfex objects for each of the SE images
        include the coadd so we get  the index right
        """
        print('loading psfex')
        desdata=os.environ['DESDATA']
        meds_desdata=self.meds_list[0]._meta['DESDATA'][0]

        psfex_lists=[]
        flags_lists=[]

        for band in self.iband:
            meds=self.meds_list[band]

            psfex_list, flags_list = self._get_psfex_objects(meds)
            psfex_lists.append( psfex_list )
            flags_lists.append( flags_list )

        return psfex_lists, flags_lists

    def _get_psfex_blacklist(self):
        """
        get the blacklist, loading if necessary
        """
        if not hasattr(self, '_psfex_blacklist'):
            fname=self['psfex_blacklist']
            blacklist_raw=read_psfex_blacklist(fname)

            blacklist={}

            for i in xrange(blacklist_raw.size):
                key='%s-%02d' % (blacklist_raw['expname'][i], blacklist_raw['ccd'][i])
                blacklist[key] = blacklist_raw['flags'][i]

            self._psfex_blacklist_raw=blacklist_raw
            self._psfex_blacklist=blacklist

        return self._psfex_blacklist

    def _psfex_path_from_image_path(self, meds, image_path):
        """
        infer the psfex path from the image path

        Mike's current flags

        1 = No stars found
        2 = Too few stars found (<50)
        4 = Too many stars found (>500)
        8 = Too high FWHM (>1.8 arcsec)
        16 = Error encountered somewhere along the line in making the PSFEx files.
        """
        desdata=os.environ['DESDATA']
        meds_desdata=meds._meta['DESDATA'][0]

        psfpath=image_path.replace('.fits.fz','_psfcat.psf')
        #print(image_path)
        #print(psfpath)

        if desdata not in psfpath:
            psfpath=psfpath.replace(meds_desdata,desdata)

        flags=0

        if self['use_psf_rerun'] and 'coadd' not in psfpath:
            psfparts=psfpath.split('/')
            psfparts[-6] = 'EXTRA' # replace 'OPS'
            psfparts[-3] = 'psfex-rerun' # replace 'red'

            psfpath='/'.join(psfparts)

            expname=psfparts[-2]
            ccd=psfparts[-1].split('_')[2]

            key='%s-%s' % (expname, ccd)

            blacklist=self._get_psfex_blacklist()
            flagsall=blacklist.get(key, 0)

            # we only worry about certain flags
            checkflags=self['psf_flags2check']
            if (flagsall & checkflags) != 0:
                flags = checkflags
                print(psfpath,flags)

        return psfpath, flags

    def _get_psfex_objects(self, meds):
        """
        Load psfex objects for all images, including coadd
        """

        psfex_list=[]
        flags_list=[]

        info=meds.get_image_info()
        nimage=info.size

        for i in xrange(nimage):
            pex=None

            impath=info['image_path'][i].strip()
            psfpath, flags = self._psfex_path_from_image_path(meds, impath)

            if flags==0:
                if not os.path.exists(psfpath):
                    # this flag is Mike's
                    # 16 = Error encountered somewhere along the line 
                    # in making the PSFEx files.
                    print("warning: missing psfex: %s" % psfpath)
                    flags = 1<<16
                else:
                    print("loading:",psfpath)
                    pex=psfex.PSFEx(psfpath)
        
            flags = flags << PSFEX_FLAGS_SHIFT
            psfex_list.append(pex)
            flags_list.append(flags)

            if flags != 0 and pex is not None:
                raise RuntimeError("got flags %d but not pex none" % flags)

        return psfex_list, flags_list

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
        res=fitter.get_result()
        dindex=self.dindex
        if res['flags']==0:
            if coadd:
                type='coadd'
            else:
                type='mb'

            print("        %s linear pars:" % type)
            print_pars(res['pars'],    front='        ')
            print_pars(res['pars_err'],front='        ')
            if 'arate' in res:
                print('        arate:',res['arate'])

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
        import fitsio
        if self.checkpoint_data is not None:
            self.data=self.checkpoint_data['data']

            # need the data to be native for the operation below
            fitsio.fitslib.array_to_native(self.data, inplace=True)

            # for nband==1 the written array drops the arrayness
            if self['nband']==1:
                raise ValueError("fix for 1 band")
            self.data.dtype=self._get_dtype()
            self.epoch_data=self.checkpoint_data['epoch_data']

            if self.epoch_data.dtype.names is not None:
                # start where we left off
                w,=numpy.where( self.epoch_data['number'] < 0)
                self.epoch_index = w.min()

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
        from .files import StagedOutFile

        print('checkpointing at',tm/60,'minutes')
        print(self.checkpoint_file)

        with StagedOutFile(self.checkpoint_file, tmpdir=self['work_dir']) as sf:
            with fitsio.FITS(sf.path,'rw',clobber=True) as fobj:
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
        return make_all_model_names(self['fit_models'], self['fit_me_galaxy'])
        '''
        models=['coadd_%s' % model for model in self['fit_models']]

        if self['fit_me_galaxy']:
            models = models + self['fit_models']

        return models
        '''

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
        dt=[('id','i8'),     # could be coadd_objects_id
            ('number','i4'), # 1-n as in sextractor
            ('band_num','i2'),
            ('cutout_index','i4'), # this is the index in meds
            ('orig_row','f8'),
            ('orig_col','f8'),
            ('file_id','i4'),   # id in meds file
            ('image_id','i8'),  # image_id specified in meds creation, e.g. for image table
            ('npix','i4'), # was not in ngmix009
            ('wsum','f8'),
            ('wmax','f8'),
            ('psf_fit_flags','i4'),
            ('psf_counts','f8'),
            ('psf_fit_g','f8',2),
            ('psf_fit_T','f8'),
            ('psf_fit_pars','f8',npars)]

        ncutout=self._count_all_cutouts()
        if ncutout > 0:
            epoch_data = numpy.zeros(ncutout, dtype=dt)

            epoch_data['id'] = DEFVAL
            epoch_data['number'] = DEFVAL
            epoch_data['band_num'] = DEFVAL
            epoch_data['cutout_index'] = DEFVAL
            epoch_data['orig_row'] = DEFVAL
            epoch_data['orig_col'] = DEFVAL
            epoch_data['file_id'] = DEFVAL
            epoch_data['image_id'] = DEFVAL
            epoch_data['psf_counts'] = DEFVAL
            epoch_data['psf_fit_g'] = PDEFVAL
            epoch_data['psf_fit_T'] = PDEFVAL
            epoch_data['psf_fit_pars'] = PDEFVAL
            epoch_data['psf_fit_flags'] = NO_ATTEMPT

            self.epoch_data=epoch_data
        else:
            self.epoch_data=numpy.zeros(1)

        # where the next psf data will be written
        self.epoch_index = 0


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
                 (n('pars_best'),'f8',np),
                 (n('pars_cov'),'f8',(np,np)),
                 (n('flux'),'f8',bshape),
                 (n('flux_cov'),'f8',fcov_shape),
                 (n('g'),'f8',2),
                 (n('g_cov'),'f8',(2,2)),
                
                 (n('s2n_w'),'f8'),
                 (n('chi2per'),'f8'),
                 (n('dof'),'f8'),
                 (n('arate'),'f8'),
                 (n('tau'),'f8'),
                ]
            
            #use a simple set for now - keep flags just in case
            dt+=[(n('flags_uw'),'i4'),
                 (n('pars_uw'),'f8',np),
                 (n('pars_cov_uw'),'f8',(np,np)),
                 (n('chi2per_uw'),'f8'),
                 (n('dof_uw'),'f8'),
                 (n('arate_uw'),'f8'),
                 (n('tau_uw'),'f8'),
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
            data[n('pars_best')] = DEFVAL
            data[n('pars_cov')] = PDEFVAL
            data[n('flux')] = DEFVAL
            data[n('flux_cov')] = PDEFVAL
            data[n('g')] = DEFVAL
            data[n('g_cov')] = PDEFVAL

            data[n('s2n_w')] = DEFVAL
            data[n('chi2per')] = PDEFVAL

            data[n('tau')] = PDEFVAL
            
            #use a simple set for now - keep flags just in case
            data[n('flags_uw')] = NO_ATTEMPT
            data[n('pars_uw')] = DEFVAL
            data[n('pars_cov_uw')] = PDEFVAL            
            data[n('chi2per_uw')] = PDEFVAL
            data[n('tau_uw')] = PDEFVAL
            
            if self['do_shear']:
                data[n('g_sens')] = DEFVAL
                data[n('P')] = DEFVAL
                data[n('Q')] = DEFVAL
                data[n('R')] = DEFVAL

     
        self.data=data


'''
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

            if self['do_shear']:
                self._add_shear_info(fitter, model)

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
                        nu=self['nu'],
                        prior=prior,
                        random_state=self.random_state)

        mhpars=self['mh_pars']
        print_pars(guess,front="    mh guess:")
        pos=fitter.run_mcmc(guess,mhpars['burnin'])
        print_pars(guess,front="    mh start after burnin:")
        pos=fitter.run_mcmc(pos,mhpars['nstep'])

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
'''

class MHMedsFitHybrid(MedsFit):
    """
    This version uses MH for fitting, with guess/steps from
    a coadd emcee run
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

        if max_s2n >= self['min_psf_s2n'] and len(self['fit_models']) > 0:
            for model in self['fit_models']:
                print('    fitting:',model)

                print('    coadd')
                self._run_model_fit(model, self['coadd_fitter_class'],coadd=True)

                if self['fit_me_galaxy']:
                    print('    multi-epoch')
                    # fitter class should be mh...
                    self._run_model_fit(model, self['fitter_class'], coadd=False)

        else:
            mess="    psf s/n too low: %s (%s)"
            mess=mess % (max_s2n,self['min_psf_s2n'])
            print(mess)
            
            flags |= LOW_PSF_FLUX

        return flags

    def _run_model_fit(self, model, fitter_type, coadd=False):
        """
        wrapper to run fit, copy pars, maybe make plots

        sets .fitter or .coadd_fitter

        this one does not currently use self['guess_type']
        """

        if coadd:
            self.guesser=self._get_guesser(self['coadd_model_guess'])
            mb_obs_list=self.sdata['coadd_mb_obs_list']
        else:
            self.guesser=self._get_guesser(self['me_model_guess'])
            mb_obs_list=self.sdata['mb_obs_list']

        fitter=self._fit_model(mb_obs_list,
                               model,
                               fitter_type=fitter_type)

        self._copy_simple_pars(fitter, coadd=coadd)

        self._print_res(fitter, coadd=coadd)

        if self['make_plots']:
            self._do_make_plots(fitter, model, coadd=coadd, fitter_type=fitter_type)

        if coadd:
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

        if self['do_shear']:
            self._add_shear_info(fitter, model)

        return fitter


    def _fit_simple_mh(self, mb_obs_list, model):
        """
        Fit one of the "simple" models, e.g. exp or dev

        use flat g prior

        first burnin is to fix the step size.  then burn in again
        and run the steps
        """

        from ngmix.fitting import MHSimple

        mhpars=self['mh_pars']

        # note flat on g!
        prior=self.gflat_priors[model]

        guess,sigmas=self.guesser(get_sigmas=True, prior=prior)

        #for olist in mb_obs_list:
        #    print("    image filename:",olist[0].filename)
        #    print("    psfex filename:",olist[0].psf.filename)

        step_sizes = 0.5*sigmas

        # this is 5-element, use 5th for all fluxes
        min_steps = mhpars['min_step_sizes']
        max_steps = 0.5*self.priors[model].get_widths()

        print_pars(max_steps, front="        max_steps:")

        for i in xrange(guess.size):
            if i > 5:
                min_step=min_steps[5]
            else:
                min_step=min_steps[i]
            max_step=max_steps[i]

            step_sizes[i] = step_sizes[i].clip(min=min_step, max=max_step)

        print_pars(step_sizes, front="        step sizes:")

        fitter=MHSimple(mb_obs_list,
                        model,
                        step_sizes,
                        prior=prior,
                        nu=self['nu'],
                        random_state=self.random_state)

        print_pars(guess,front="        mh guess:   ")
        pos=fitter.run_mcmc(guess,mhpars['burnin'])

        n=int(mhpars['burnin']*0.1)

        acc=fitter.sampler.get_accepted()
        arate = acc[-n:].sum()/(1.0*n)
        print("        arate of last",n,"is",arate)

        if arate < 0.01:
            fac=0.01/0.5
        else:
            fac = arate/0.5

        pos=fitter.get_best_pars()
        print_pars(pos,        front="            mh start after 1st burnin:")

        step_sizes *= fac
        fitter.set_step_sizes(step_sizes)
        print_pars(step_sizes, front="            new step sizes:")

        pos=fitter.run_mcmc(pos,mhpars['burnin'])

        # in case we ended on a bad point
        pos=fitter.get_best_pars()
        print_pars(pos,        front="            mh start after 2nd burnin:")
        pos=fitter.run_mcmc(pos,mhpars['nstep'])

        return fitter

class MHMedsFitModelNbrs(MHMedsFitHybrid):
    """
    Models Nbrs
    """

    def _fit_all_models(self):
        """
        Fit psf flux and other models
        """
        
        #default to true just in case
        if 'fit_coadd_galaxy' not in self:
            self['fit_coadd_galaxy'] = True
        
        mindex_local = self.mindex #index in current meds file
        meds = self.meds_list[0]
        number = meds['number'][mindex_local] #number for seg map, index+1 into entire meds file
        mindex_global = number-1
        
        flags=0
        # fit both coadd and se psf flux if exists
        self._fit_psf_flux()

        dindex=self.dindex
        s2n=self.data['coadd_psf_flux'][dindex,:]/self.data['coadd_psf_flux_err'][dindex,:]
        max_s2n=numpy.nanmax(s2n)

        if max_s2n >= self['min_psf_s2n'] and len(self['fit_models']) > 0:
            for model in self['fit_models']:
                print('    fitting:',model)
                
                if self['fit_coadd_galaxy']:
                    print('    coadd')
                    self._run_model_fit(model, self['coadd_fitter_class'], mindex_global, coadd=True)

                if self['fit_me_galaxy']:
                    print('    multi-epoch')
                    # fitter class should be mh...
                    self._run_model_fit(model, self['fitter_class'], mindex_global, coadd=False)

        else:
            mess="    psf s/n too low: %s (%s)"
            mess=mess % (max_s2n,self['min_psf_s2n'])
            print(mess)
            
            flags |= LOW_PSF_FLUX

        return flags

    def _run_model_fit(self, model, fitter_type, mindex_global, coadd=False):
        """
        wrapper to run fit, copy pars, maybe make plots

        sets .fitter or .coadd_fitter

        this one does not currently use self['guess_type']
        """
        
        nmod = Namer(model)        
        if coadd:
            n = Namer('coadd')
            pars = self.model_data['model_fits'][n(nmod('pars_best'))][mindex_global]
            pars_cov = self.model_data['model_fits'][n(nmod('pars_cov'))][mindex_global]
            pars_err = numpy.array([numpy.sqrt(pars_cov[i,i]) for i in xrange(len(pars))])
            self.guesser=FromFullParsGuesser(pars,pars_err,scaling=None)
            mb_obs_list=self.sdata['coadd_mb_obs_list']
        else:
            pars = self.model_data['model_fits'][nmod('pars_best')][mindex_global]
            pars_cov = self.model_data['model_fits'][nmod('pars_cov')][mindex_global]
            pars_err = numpy.array([numpy.sqrt(pars_cov[i,i]) for i in xrange(len(pars))])
            self.guesser=FromFullParsGuesser(pars,pars_err,scaling=None)
            mb_obs_list=self.sdata['mb_obs_list']

        fitter=self._fit_model(mb_obs_list,
                               model,
                               fitter_type=fitter_type)

        self._copy_simple_pars(fitter, coadd=coadd)

        self._print_res(fitter, coadd=coadd)

        if self['make_plots']:
            self._do_make_plots(fitter, model, coadd=coadd, fitter_type=fitter_type)

        if coadd:
            self.coadd_fitter=fitter
        else:
            self.fitter=fitter



class MHMedsFitHybridIter(MHMedsFitHybrid):
    """
    This version uses MH for fitting, with guess/steps from
    a coadd emcee run, which is seeded via iterating between 
    a direct maximizer and emcee run on the coadd
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
        
        if max_s2n >= self['min_psf_s2n'] and len(self['fit_models']) > 0:
            for model in self['fit_models']:
                print('    fitting:',model)
                
                fmt = "%.6f "*(5+self['nband'])
                print('    coadd iter fit')
                print('        using method \'%s\' for minimizer' % self['coadd_iter']['min_method'])
                
                self.coadd_guesser = \
                    self._guess_params_iter(self, 
                                            self.sdata['coadd_mb_obs_list'], 
                                            model, 
                                            self['coadd_iter'],
                                            self._get_guesser('coadd_psf'))
                if self.coadd_guesser == None:
                    self.coadd_guesser = self._get_guesser('coadd_psf')
                
                print('    coadd')                
                self._run_model_fit(model, self['coadd_fitter_class'],coadd=True)

                if self['fit_me_galaxy']:
                    print('    multi-epoch')
                    if 'me_iter' in self:
                        self.me_guesser = \
                            self._guess_params_iter(self,
                                                    self.sdata['mb_obs_list'],
                                                    model, self['me_iter'], 
                                                    self._get_guesser('coadd_psf'))
                    else:
                        self.me_guesser = None
                    if self.me_guesser == None:
                        self.me_guesser = self._get_guesser('coadd_psf')
                    # fitter class should be mh...
                    self._run_model_fit(model, self['fitter_class'], coadd=False)
        else:
            mess="    psf s/n too low: %s (%s)"
            mess=mess % (max_s2n,self['min_psf_s2n'])
            print(mess)
            
            flags |= LOW_PSF_FLUX

        return flags

    def _guess_params_iter(self, mb_obs_list, model, params, start):
        for i in xrange(niter):
            print('        iter % 3d of %d'%(i+1,niter))
            if i == 0:
                self.guesser = start
            emceefit = self._fit_simple_emcee_guess(mb_obs_list, model, params)
            pars = emceefit.get_best_pars()
            bestlk = numpy.max(emceefit.get_lnprobs())
            print('            emcee:',fmt%tuple(pars),'loglike = %lf'%bestlk)
            self.guesser = FixedParsGuesser(pars,pars*0.1) #making that up, but it doesn't matter                    
            if params['min_method'] == 'lm':
                greedyfit = self._fit_simple_lm(mb_obs_list, model, params)
            else:
                greedyfit = self._fit_simple_max(mb_obs_list, model, params)
            pars = greedyfit._result['pars']
            if 'pars_err' in greedyfit._result:
                pars_err = greedyfit._result['pars_err']
            else:
                pars_err = greedyfit._result['pars']*0.05
            bestlk = greedyfit.calc_lnprob(pars)
            print('            min:  ',fmt%tuple(pars),'loglike = %lf'%bestlk)
            self.guesser = FromAlmostFullParsGuesser(pars,pars_err,scaling=None)
        
        if numpy.all(numpy.abs(pars) < 1e7):
            return self.guesser
        else:
            return None
    
    def _run_model_fit(self, model, fitter_type, coadd=False):
        """
        wrapper to run fit, copy pars, maybe make plots

        sets .fitter or .coadd_fitter

        this one does not currently use self['guess_type']
        """

        if coadd:
            self.guesser=self.coadd_guesser
            mb_obs_list=self.sdata['coadd_mb_obs_list']
        else:
            self.guesser=self.me_guesser
            mb_obs_list=self.sdata['mb_obs_list']

        fitter=self._fit_model(mb_obs_list,
                               model,
                               fitter_type=fitter_type)

        self._copy_simple_pars(fitter, coadd=coadd)

        self._print_res(fitter, coadd=coadd)

        if self['make_plots']:
            self._do_make_plots(fitter, model, coadd=coadd, fitter_type=fitter_type)

        if coadd:
            self.coadd_fitter=fitter
        else:
            self.fitter=fitter

    def _fit_simple_emcee_guess(self, mb_obs_list, model, params):
        """
        Fit one of the "simple" models, e.g. exp or dev

        use flat g prior
        """

        from ngmix.fitting import MCMCSimple

        # note flat on g!
        prior=self.gflat_priors[model]

        epars=params['emcee_pars']
        guess=self.guesser(n=epars['nwalkers'], prior=prior)
        #for olist in mb_obs_list:
        #    print("    image filename:",olist[0].filename)
        #    print("    psfex filename:",olist[0].psf.filename)

        fitter=MCMCSimple(mb_obs_list,
                          model,
                          nu=self['nu'],
                          prior=prior,
                          nwalkers=epars['nwalkers'],
                          mca_a=epars['a'],
                          random_state=self.random_state)

        pos=fitter.run_mcmc(guess,epars['burnin'])
        pos=fitter.run_mcmc(pos,epars['nstep'])

        return fitter

    def _fit_simple_lm(self, mb_obs_list, model, params):
        """
        Fit one of the "simple" models, e.g. exp or dev

        use flat g prior
        """

        from ngmix.fitting import LMSimple

        # note the prior is *not* flat in g! This is important to constrain the
        # covariance matrix
        
        prior=self.priors[model]

        ntry=params['lm_ntry']
        for i in xrange(ntry):
            guess=self.guesser(prior=prior)
            #print_pars(guess, front='            lm guess:')

            fitter=LMSimple(mb_obs_list,
                            model,
                            prior=prior,
                            lm_pars=params['lm_pars'])

            fitter.run_lm(guess)
            res=fitter.get_result()
            if res['flags']==0:
                break

        res['ntry']=i+1
        return fitter

    def _fit_simple_max(self, mb_obs_list, model, params):
        from ngmix.fitting import MaxSimple        
        guess=self.guesser(prior=self.priors[model])
        fitter=MaxSimple(mb_obs_list,model,method=params['min_method'])
        fitter.run_max(guess)
        return fitter
                            


class GuesserBase(object):
    def _fix_guess(self, guess, prior, ntry=4):
        from ngmix.priors import LOWVAL

        #guess[:,2]=-9999
        n=guess.shape[0]
        for j in xrange(n):
            for itry in xrange(ntry):

                try:
                    lnp=prior.get_lnprob_scalar(guess[j,:])

                    if lnp <= LOWVAL:
                        dosample=True
                    else:
                        dosample=False
                except GMixRangeError as err:
                    dosample=True

                if dosample:
                    print_pars(guess[j,:], front="bad guess:")
                    if itry < ntry:
                        guess[j,:] = prior.sample()
                    else:
                        raise UtterFailure("could not find a good guess")
                else:
                    break


class FromMCMCGuesser(GuesserBase):
    """
    get guesses from a set of trials
    """
    def __init__(self, trials, sigmas):
        self.trials=trials
        self.sigmas=sigmas
        self.npars=trials.shape[1]

        #self.lnprobs=lnprobs
        #self.lnp_sort=lnprobs.argsort()

    def __call__(self, n=None, get_sigmas=False, prior=None):
        """
        get a random sample from the best points
        """
        import random

        if n is None:
            is_scalar=True
            n=1
        else:
            is_scalar=False

        # choose randomly from best
        #indices = self.lnp_sort[-n:]
        #guess = self.trials[indices, :]

        trials=self.trials
        np = trials.shape[0]

        rand_int = random.sample(xrange(np), n)
        guess=trials[rand_int, :]

        if prior is not None:
            self._fix_guess(guess, prior)

        w,=numpy.where(guess[:,4] <= 0.0)
        if w.size > 0:
            guess[w,4] = 0.05*srandu(w.size)

        for i in xrange(5, self.npars):
            w,=numpy.where(guess[:,i] <= 0.0)
            if w.size > 0:
                guess[w,i] = (1.0 + 0.1*srandu(w.size))

        #print("guess from mcmc:")
        #for i in xrange(n):
        #    print_pars(guess[i,:], front="%d: " % i)

        if is_scalar:
            guess=guess[0,:]

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
    def __init__(self, T, fluxes, scaling='linear'):
        self.T=T
        self.fluxes=fluxes
        self.scaling=scaling

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

        if self.scaling=='linear':
            if self.T <= 0.0:
                guess[:,4] = 0.05*srandu(n)
            else:
                guess[:,4] = self.T*(1.0 + 0.1*srandu(n))

            fluxes=self.fluxes
            for band in xrange(nband):
                if fluxes[band] <= 0.0:
                    guess[:,5+band] = (1.0 + 0.1*srandu(n))
                else:
                    guess[:,5+band] = fluxes[band]*(1.0 + 0.1*srandu(n))

        else:
            guess[:,4] = self.log_T + 0.1*srandu(n)

            for band in xrange(nband):
                guess[:,5+band] = self.log_fluxes[band] + 0.1*srandu(n)

        if prior is not None:
            self._fix_guess(guess, prior)

        if n==1:
            guess=guess[0,:]
        return guess

class FixedParsGuesser(GuesserBase):
    """
    just return a copy of the input pars
    """
    def __init__(self, pars, pars_err):
        self.pars=pars
        self.pars_err=pars_err

    def __call__(self, get_sigmas=False, prior=None):
        """
        center, shape are just distributed around zero
        """

        guess=self.pars.copy()
        if get_sigmas:
            return guess, self.pars_err
        else:
            return guess


class FromParsGuesser(GuesserBase):
    """
    get full guesses from just T,fluxes associated with
    psf
    """
    def __init__(self, pars, pars_err, scaling='linear'):
        self.pars=pars
        self.pars_err=pars_err
        self.scaling=scaling

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

        width = pars*0 + 0.1

        guess=numpy.zeros( (n, npars) )

        guess[:,0] = width[0]*srandu(n)
        guess[:,1] = width[1]*srandu(n)

        guess_shape=get_shape_guess(pars[2],pars[3],n,width[2:2+2])
        guess[:,2]=guess_shape[:,0]
        guess[:,3]=guess_shape[:,1]

        for i in xrange(4,npars):
            if self.scaling=='linear':
                if pars[i] <= 0.0:
                    guess[:,i] = width[i]*srandu(n)
                else:
                    guess[:,i] = pars[i]*(1.0 + width[i]*srandu(n))
            else:
                # we add to log pars!
                guess[:,i] = pars[i] + width[i]*srandu(n)

        if prior is not None:
            self._fix_guess(guess, prior)

        if is_scalar:
            guess=guess[0,:]

        if get_sigmas:
            return guess, self.pars_err
        else:
            return guess


class FromAlmostFullParsGuesser(GuesserBase):
    """
    get full guesses from just g1,g2,T,fluxes associated with
    psf
    """
    def __init__(self, pars, pars_err, scaling='linear'):
        self.pars=pars
        self.pars_err=pars_err
        self.scaling=scaling

    def __call__(self, n=None, get_sigmas=False, prior=None):
        """
        center is just distributed around zero
        """

        if n is None:
            n=1
            is_scalar=True
        else:
            is_scalar=False

        pars=self.pars
        npars=pars.size

        width = pars*0 + 0.1

        guess=numpy.zeros( (n, npars) )

        guess[:,0] = width[0]*srandu(n)
        guess[:,1] = width[1]*srandu(n)

        for j in xrange(n):
            itr = 0
            maxitr = 100
            while itr < maxitr:
                for i in xrange(2,npars):
                    if self.scaling=='linear':
                        if pars[i] <= 0.0:
                            guess[j,:] = width[i]*srandu(1)
                        else:
                            guess[j,i] = pars[i]*(1.0 + width[i]*srandu(1))
                    else:
                        # we add to log pars!
                        guess[j,i] = pars[i] + width[i]*srandu(1)

                if numpy.abs(guess[j,2]) < 1.0 \
                        and numpy.abs(guess[j,3]) < 1.0 \
                        and guess[j,2]*guess[j,2] + guess[j,3]*guess[j,3] < 1.0:
                    break
                itr += 1

        if prior is not None:
            self._fix_guess(guess, prior)

        if is_scalar:
            guess=guess[0,:]

        if get_sigmas:
            return guess, self.pars_err
        else:
            return guess


class FromFullParsGuesser(GuesserBase):
    """
    get full guesses
    """
    def __init__(self, pars, pars_err, scaling='linear'):
        self.pars=pars
        self.pars_err=pars_err
        self.scaling=scaling

    def __call__(self, n=None, get_sigmas=False, prior=None):
        if n is None:
            n=1
            is_scalar=True
        else:
            is_scalar=False

        pars=self.pars
        npars=pars.size

        width = pars*0 + 0.1

        guess=numpy.zeros( (n, npars) )

        for j in xrange(n):
            itr = 0
            maxitr = 100
            while itr < maxitr:
                for i in xrange(npars):
                    if self.scaling=='linear':
                        if pars[i] <= 0.0:
                            guess[j,:] = width[i]*srandu(1)
                        else:
                            guess[j,i] = pars[i]*(1.0 + width[i]*srandu(1))
                    else:
                        # we add to log pars!
                        guess[j,i] = pars[i] + width[i]*srandu(1)

                if numpy.abs(guess[j,2]) < 1.0 \
                        and numpy.abs(guess[j,3]) < 1.0 \
                        and guess[j,2]*guess[j,2] + guess[j,3]*guess[j,3] < 1.0:
                    break
                itr += 1

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
             'dof']


_psf_ngauss_map={'em1':1, 'em2':2}
def get_psf_ngauss(psf_model):
    if psf_model not in _psf_ngauss_map:
        raise ValueError("bad psf model: '%s'" % psf_model)
    return _psf_ngauss_map[psf_model]


def get_as_list(data_in):
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


def get_coadd_cat_path(image_path):
    cat_path=image_path.replace('.fits.fz','').replace('.fits','')

    cat_path='%s_cat.fits' % cat_path

    return cat_path

def read_psfex_blacklist(fname):
    from esutil import recfile

    print("reading psfex blacklist from:", fname)

    dt=[('run','S23'),
        ('expname','S14'),
        ('ccd','i2'),
        ('flags','i4')]
    
    with recfile.Recfile(fname, 'r', delim=' ', dtype=dt) as robj:
        data=robj.read()

    return data

class Namer(object):
    def __init__(self, front=None):
        self.front=front
    def __call__(self, name):
        if self.front is None or self.front=='':
            return name
        else:
            return '%s_%s' % (self.front, name)

def make_all_model_names(fit_models, fit_me_galaxy):
    """
    get all model names, includeing the coadd_ ones
    """
    models=['coadd_%s' % model for model in fit_models]

    if fit_me_galaxy:
        models = models + fit_models

    return models


