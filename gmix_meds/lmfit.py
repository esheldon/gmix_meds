import os
from sys import stderr
import time
import numpy
from numpy import sqrt, diag
from numpy.random import randn
import fitsio
import meds

try:
    import gmix_image
    from gmix_image.gmix_fit import LM_MAX_TRY, \
            GMixFitPSFJacob,\
            GMixFitMultiSimple,GMixFitMultiCModel, \
            GMixFitMultiPSFFlux,GMixFitMultiMatch
    from gmix_image.util import print_pars, srandu
    from gmix_image.gmix_em import GMixEMBoot
    from gmix_image.priors import CenPrior
except:
    print 'could not import gmix_image'

import psfex

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

ALGO_TIMEOUT=2**6

PSF1_NOT_KEPT=2**7

NO_ATTEMPT=2**30

PSF_S2N=1.e6
PSF_OFFSET_MAX=0.25

SIMPLE_MODELS_DEFAULT = ['exp','dev']

_psf_ngauss_map={'lm1':1, 'lm2':2, 'lm3':3,
                 'em1':1, 'em2':2, 'em3':3}
def get_psf_ngauss(psf_model):
    if psf_model not in _psf_ngauss_map:
        raise ValueError("bad psf model: '%s'" % psf_model)
    return _psf_ngauss_map[psf_model]

class MedsFit(object):
    def __init__(self, meds_file, **keys):
        """
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

        self.meds_file=meds_file
        self.checkpoint = self.conf.get('checkpoint',172800)
        self.checkpoint_file = self.conf.get('checkpoint_file',None)
        self._checkpoint_data=keys.get('checkpoint_data',None)

        print >>stderr,'meds file:',meds_file
        self.meds=meds.MEDS(meds_file)
        self.meds_meta=self.meds.get_meta()
        self.nobj=self.meds.size

        # in arcsec (or units of jacobian)
        self.use_cenprior=keys.get("use_cenprior",True)
        self.cen_width=keys.get('cen_width',1.0)

        self.gprior=keys.get('gprior',None)

        self.obj_range=keys.get('obj_range',None)

        self.psf_model=keys.get('psf_model','em2')
        self.psf_offset_max=keys.get('psf_offset_max',PSF_OFFSET_MAX)
        self.psf_ngauss=get_psf_ngauss(self.psf_model)

        self.debug=keys.get('debug',0)

        self.psf_ntry=keys.get('psf_ntry', LM_MAX_TRY)
        self.obj_ntry=keys.get('obj_ntry',2)

        self.region=keys.get('region','seg_and_sky')
        self.max_box_size=keys.get('max_box_size',2048)

        self.simple_models=keys.get('simple_models',SIMPLE_MODELS_DEFAULT )

        self.match_use_band_center = keys.get('match_use_band_center',False)
        self.match_self = keys.get('match_self',False)
        self.reject_outliers=keys.get('reject_outliers',False)

        self.make_plots=keys.get('make_plots',False)
        self.prompt=keys.get('prompt',True)

        self._set_index_list()
        det_cat=keys.get('det_cat',None)
        self._set_det_cat_and_struct(det_cat)

        self.psfex_list = self._get_all_psfex_objects(self.meds)

    def get_data(self):
        """
        Get the data structure.  If a subset was requested, only those rows are
        returned.
        """
        return self.data

    def get_meds_meta(self):
        return self.meds_meta.copy()

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
        self.checkpointed=False

        for dindex in xrange(num):
            if self.data['processed'][dindex]==1:
                # checkpointing
                continue

            mindex = self.index_list[dindex]
            print >>stderr,'index: %d:%d' % (mindex,last),
            self._fit_obj(dindex)

            tm=time.time()-t0

            if self._should_checkpoint(tm):
                self._write_checkpoint(tm)

        tm=time.time()-t0
        print >>stderr,"time per:",tm/num
    
    def _should_checkpoint(self, tm):
        if (tm > self.checkpoint
                and self.checkpoint_file is not None
                and not self.checkpointed):
            return True
        else:
            return False

    def _fit_obj(self, dindex):
        """
        Process the indicated object

        The first cutout is always the coadd, followed by
        the SE images which will be fit simultaneously
        """

        t0=time.time()

        # for checkpointing
        self.data['processed'][dindex]=1

        mindex=self.index_list[dindex]

        self.data['id'][dindex] = self.meds['number'][mindex]

        self.data['flags'][dindex] = self._obj_check(self.meds, mindex)
        if self.data['flags'][dindex] != 0:
            return 0

        imlist0,wtlist0,self.coadd = self._get_imlist_wtlist(self.meds,mindex)
        jacob_list0=self._get_jacobian_list(self.meds,mindex)


        self.data['nimage_tot'][dindex] = len(imlist0)
        print >>stderr,imlist0[0].shape
    
        keep_list,psf_gmix_list=self._fit_psfs(self.meds,mindex,jacob_list0,self.psfex_list)
        if len(psf_gmix_list)==0:
            self.data['flags'][dindex] |= PSF_FIT_FAILURE
            return

        keep_list,psf_gmix_list=self._remove_bad_psfs(keep_list,psf_gmix_list)
        if len(psf_gmix_list)==0:
            self.data['flags'][dindex] |= PSF_LARGE_OFFSETS
            return

        imlist = [imlist0[i] for i in keep_list]
        wtlist = [wtlist0[i] for i in keep_list]
        jacob_list = [jacob_list0[i] for i in keep_list]

        self.data['nimage_use'][dindex] = len(imlist)

        sdata={'keep_list':keep_list,
               'imlist':imlist,
               'wtlist':wtlist,
               'jacob_list':jacob_list,
               'psf_gmix_list':psf_gmix_list}

        self._do_all_fits(dindex, sdata)

        self.data['time'][dindex] = time.time()-t0

    def _do_all_fits(self, dindex, sdata):

        if 'psf' in self.conf['fit_types']:
            self._fit_psf_flux(dindex, sdata)
        else:
            raise ValueError("you should do a psf_flux fit")

        psf_s2n=self.data['psf_flux_s2n'][dindex]
        if psf_s2n >= self.conf['min_psf_s2n']:
            if 'simple' in self.conf['fit_types']:
                self._fit_simple_models(dindex, sdata)
            if 'cmodel' in self.conf['fit_types']:
                self._fit_cmodel(dindex, sdata)
            if 'match' in self.conf['fit_types']:
                self._fit_match(dindex, sdata)
        else:
            mess="    psf s/n too low: %s (%s)"
            mess=mess % (psf_s2n,self.conf['min_psf_s2n'])
            print >>stderr,mess

        if self.debug >= 3:
            self._debug_image(sdata['imlist'][0],sdata['wtlist'][-1])



    def _obj_check(self, meds, mindex):
        flags=0

        box_size=meds['box_size'][mindex]
        if box_size > self.max_box_size:
            print >>stderr,'Box size too big:',box_size
            flags |= BOX_SIZE_TOO_BIG

        if meds['ncutout'][mindex] < 2:
            print >>stderr,'No SE cutouts'
            flags |= NO_SE_CUTOUTS
        return flags

    def _fit_psfs(self,meds,mindex,jacob_list,psfex_list):
        """
        Generate psfex images for all SE images and fit
        them to gaussian mixture models
        """
        ptuple = self._get_psfex_reclist(meds, psfex_list, mindex)
        imlist,ivarlist,cenlist,siglist,flist,cenpix=ptuple

        keep_list=[]
        gmix_list=[]

        for i in xrange(len(imlist)):
            im=imlist[i]
            ivar=ivarlist[i]
            jacob0=jacob_list[i]
            sigma=siglist[i]

            cen0=cenlist[i]
            # the dimensions of the psfs are different, need
            # new center
            jacob={}
            jacob.update(jacob0)
            jacob['row0'] = cen0[0]
            jacob['col0'] = cen0[1]

            gm=self._do_fit_psf(im,jacob,ivar,sigma)

            res=gm.get_result()
            if res['flags'] == 0:
                
                keep_list.append(i)
                gmix_psf=gm.get_gmix()
                gmix_list.append( gmix_psf )

                if False:
                    self._compare_psf_model(im, gm, mindex, i, flist[i], cenpix)
            else:
                print >>stderr,'psf fail',flist[i]


        return keep_list, gmix_list

    def _remove_bad_psfs(self, keep_list0, gmix_list0):
        from .double_psf import calc_offset_arcsec

        if self.psf_model != 'em2':
            return keep_list0, gmix_list0

        keep_list=[]
        gmix_list=[]
        for i in xrange(len(gmix_list0)):
            gmix=gmix_list0[i]
            ki=keep_list0[i]

            offset_arcsec = calc_offset_arcsec(gmix)
            if offset_arcsec < self.psf_offset_max:
                keep_list.append( ki )
                gmix_list.append( gmix )
            else:
                print >>stderr,'    removed offset psf:',offset_arcsec

        return keep_list, gmix_list

    def _do_fit_psf(self, im, jacob, ivar, sigma_guess):
        if 'lm' in self.psf_model:
            gm=self._run_psf_lm_fit(im,ivar,jacob)
        elif 'em' in self.psf_model:
            gm=self._run_psf_em_fit(im,ivar,jacob,sigma_guess)
        else:
            raise RuntimeError("bad psf model '%s'" % self.psf_model)
        return gm

    def _run_psf_lm_fit(self, im, ivar, jacob):
        cen_prior=None
        if self.use_cenprior:
            cen_prior=CenPrior([0.0]*2, [self.cen_width]*2)

        gm=GMixFitPSFJacob(im,
                           ivar,
                           jacob,
                           self.psf_ngauss,
                           cen_prior=self.cen_prior,
                           lm_max_try=self.psf_ntry)
        return gm

    def _run_psf_em_fit(self, im, ivar, jacob, sigma_guess):
        cen_guess=[0.0, 0.0]
        gm=GMixEMBoot(im, self.psf_ngauss, cen_guess,
                      sigma_guess=sigma_guess,
                      jacobian=jacob,
                      ivar=ivar,
                      maxtry=self.psf_ntry)
        return gm


    def _compare_psf_model(self, im, gm, mindex, i,fname,cenpix):
        """
        Since we work in sky coords, can only generate the
        diff image currently
        """
        import os
        import images

        print fname
        name='%s_%06d_%02d' % (self.psf_model,mindex,i)

        imsum=im.sum()
        model=gm.get_model()
        model *= imsum/model.sum()

        if 'em2' in self.psf_model:
            gmix = gm.get_gmix()
            gl=gmix.get_dlist()
            offset=sqrt( (gl[0]['row']-gl[1]['row'])**2 + 
                         (gl[0]['col']-gl[1]['col'])**2 )
            print >>stderr,'offset:',offset
            for g in gl:
                print >>stderr,'gauss %d:' % (i+1),g['row'],g['col']

        #diff = model-im
        #resid = sqrt( (diff**2).sum() )/imsum

        #title='%s sq resid: %s' % (name,resid)
        #plt=images.view(diff, title=title,show=False)

        bname=os.path.basename(fname)
        bname=bname.replace('_psfcat.psf','')

        title='%s row: %.1f col: %.2f' % (bname,cenpix[0],cenpix[1])
        plt=images.compare_images(im, model,show=False,title=title)

        plt.title_style['fontsize']=2

        d=os.path.join(os.environ['HOME'], 'tmp','test-psf-rec')
        if not os.path.exists(d):
            os.makedirs(d)

        pngname='%s_%s.png' % (bname,name)
        path=os.path.join(d,pngname)
        print >>stderr,'    ',path
        plt.write_img(1000,1000,path)





    def _fit_simple_models(self, dindex, sdata):
        """
        Fit all the simple models
        """

        pars_guess=None
        for model in self.simple_models:
            print >>stderr,'    fitting:',model

            gm=self._fit_simple(dindex, model, sdata)

            res=gm.get_result()
            rfc_res=gm.get_rfc_result()

            n=get_model_names(model)

            if self.debug:
                self._print_simple_stats(n, rfc_res, res)

            self._copy_simple_pars(dindex, rfc_res, res, n )


    def _fit_simple(self, dindex, model, sdata):
        """
        Fit one of the "simple" models, e.g. exp or dev
        """

        if self.data['psf_flags'][dindex]==0:
            T_guess = 16.0
            counts_guess=2*self.data['psf_flux'][dindex]
        else:
            T_guess=None
            counts_guess=None

        cen_prior=None
        if self.use_cenprior:
            cen_prior=self._get_simple_cen_prior(dindex)


        gm=GMixFitMultiSimple(sdata['imlist'],
                              sdata['wtlist'],
                              sdata['jacob_list'],
                              sdata['psf_gmix_list'],
                              model,
                              cen_prior=cen_prior,
                              gprior=self.gprior,
                              lm_max_try=self.obj_ntry,
                              T_guess=T_guess,
                              counts_guess=counts_guess)

        return gm

    def _fit_simple_many_tries(self, dindex, model, sdata):
        """
        Fit one of the "simple" models, e.g. exp or dev
        """

        if self.data['psf_flags'][dindex]==0:
            T_guess = 16.0
            counts_guess=2*self.data['psf_flux'][dindex]
        else:
            T_guess=None
            counts_guess=None

        cen_prior=None
        if self.use_cenprior:
            cen_prior=self._get_simple_cen_prior(dindex)

        dim=sdata['imlist'][0].shape[0]

        goal=5
        if dim <= 128:
            ntries=20
        else:
            ntries=10

        gmlist=[]
        llist=[]
        for ipass in xrange(ntries):

            gm0=GMixFitMultiSimple(sdata['imlist'],
                                   sdata['wtlist'],
                                   sdata['jacob_list'],
                                   sdata['psf_gmix_list'],
                                   model,
                                   cen_prior=cen_prior,
                                   gprior=self.gprior,
                                   lm_max_try=self.obj_ntry,
                                   T_guess=T_guess,
                                   counts_guess=counts_guess)
            res=gm0.get_result()
            if res['flags']!=0:
                continue

            gmlist.append(gm0)
            llist.append(res['loglike'])

            T_guess=res['pars'][4]
            counts_guess=res['pars'][5]

            if counts_guess < 0:
                counts_guess=0
            elif abs(counts_guess > 1.e6):
                counts_guess=100
            if T_guess < 0:
                T_guess=8
            elif abs(T_guess > 1000):
                T_guess=100

            if len(gmlist) == goal:
                break
            
        print >>stderr,'        ngood:',len(gmlist)
        if len(gmlist)==0:
            return gm0

        llist=numpy.array(llist)
        ibest = llist.argmax()
        gm = gmlist[ibest]
        return gm

    def _get_simple_cen_prior(self,dindex):
        if self.data['psf_flags'][dindex]==0:
            cen=self.data['psf_pars'][dindex,0:0+2]
        else:
            cen=[0.0,0.0]

        cen_prior=CenPrior(cen, [self.cen_width]*2)
        return cen_prior

    def _copy_simple_pars(self, dindex, rfc_res, res, n):
        if rfc_res is not None:
            self.data[n['rfc_flags']][dindex] = rfc_res['flags']
            self.data[n['rfc_iter']][dindex] = rfc_res['numiter']
            self.data[n['rfc_tries']][dindex] = rfc_res['ntry']

            if rfc_res['flags']==0:
                self.data[n['rfc_pars']][dindex,:] = rfc_res['pars']
                self.data[n['rfc_pars_cov']][dindex,:] = rfc_res['pcov']

        self.data[n['flags']][dindex] = res['flags']
        self.data[n['iter']][dindex] = res['numiter']
        self.data[n['tries']][dindex] = res['ntry']

        if res['flags'] == 0:
            self.data[n['pars']][dindex,:] = res['pars']
            self.data[n['pars_cov']][dindex,:,:] = res['pcov']

            flux=res['pars'][5]
            flux_err=sqrt(res['pcov'][5,5])
            self.data[n['flux']][dindex] = flux
            self.data[n['flux_err']][dindex] = flux_err

            self.data[n['g']][dindex,:] = res['pars'][2:2+2]
            self.data[n['g_cov']][dindex,:,:] = res['pcov'][2:2+2,2:2+2]

            for sn in _stat_names:
                self.data[n[sn]][dindex] = res[sn]
        else:
            if self.debug:
                print >>stderr,'flags != 0, errmsg:',res['errmsg']
            if self.debug > 1 and self.debug < 3:
                self._debug_image(sdata['imlist'][0],sdata['wtlist'][0])



    def _fit_cmodel(self, dindex, sdata):
        if self.debug:
            print >>stderr,'\tfitting frac_dev'

        self.data['cmodel_flags'][dindex]=0

        if self.data['exp_flags'][dindex]!=0:
            self.data['cmodel_flags'][dindex] |= EXP_FIT_FAILURE
        if self.data['dev_flags'][dindex]!=0:
            self.data['cmodel_flags'][dindex] |= DEV_FIT_FAILURE

        if self.data['cmodel_flags'][dindex] != 0:
            return

        exp_gmix = gmix_image.GMix(self.data['exp_pars'][dindex],type='exp')
        dev_gmix = gmix_image.GMix(self.data['dev_pars'][dindex],type='dev')

        if (self.data['exp_loglike'][dindex] 
            > self.data['dev_loglike'][dindex]):
            fracdev_start=0.1
        else:
            fracdev_start=0.9

        gm=GMixFitMultiCModel(sdata['imlist'],
                              sdata['wtlist'],
                              sdata['jacob_list'],
                              sdata['psf_gmix_list'],
                              exp_gmix,
                              dev_gmix,
                              fracdev_start,
                              lm_max_try=self.obj_ntry)
        res=gm.get_result()
        self.data['cmodel_flags'][dindex] = res['flags']
        self.data['cmodel_iter'][dindex] = res['numiter']
        self.data['cmodel_tries'][dindex] = res['ntry']

        if res['flags']==0:
            f=res['fracdev']
            ferr=res['fracdev_err']
            self.data['frac_dev'][dindex] = f
            self.data['frac_dev_err'][dindex] = ferr
            flux=(1.-f)*self.data['exp_flux'][dindex] \
                    + f*self.data['dev_flux'][dindex]
            flux_err2=(1.-f)**2*self.data['exp_flux_err'][dindex]**2 \
                         + f**2*self.data['dev_flux_err'][dindex]**2
            flux_err=sqrt(flux_err2)
            self.data['cmodel_flux'][dindex] = flux
            self.data['cmodel_flux_err'][dindex] = flux_err

            if self.debug:
                fmt='\t\t%s: %g +/- %g'
                print >>stderr,fmt % ('frac_dev',f,ferr)
                print >>stderr,fmt % ('cmodel_flux',flux,flux_err)
                    
    def _fit_psf_flux(self, dindex, sdata):
        if self.debug:
            print >>stderr,'\tfitting psf flux'

        cen_prior=None
        if self.use_cenprior:
            cen_prior=CenPrior([0.0]*2, [self.cen_width]*2)

        gm=GMixFitMultiPSFFlux(sdata['imlist'],
                               sdata['wtlist'],
                               sdata['jacob_list'],
                               sdata['psf_gmix_list'],
                               cen_prior=cen_prior,
                               lm_max_try=self.obj_ntry)
        res=gm.get_result()
        self.data['psf_flags'][dindex] = res['flags']
        self.data['psf_iter'][dindex] = res['numiter']
        self.data['psf_tries'][dindex] = res['ntry']

        if res['flags']==0:
            self.data['psf_pars'][dindex,:]=res['pars']
            self.data['psf_pars_cov'][dindex,:,:] = res['pcov']

            flux=res['pars'][2]
            flux_err=sqrt(res['pcov'][2,2])
            self.data['psf_flux'][dindex] = flux
            self.data['psf_flux_err'][dindex] = flux_err
            self.data['psf_flux_s2n'][dindex] = flux/flux_err

            print >>stderr,'    psf_flux: %g +/- %g' % (flux,flux_err)

            n=get_model_names('psf')
            for sn in _stat_names:
                self.data[n[sn]][dindex] = res[sn]

            if self.debug:
                fmt='\t\t%s: %g +/- %g'
                print >>stderr,fmt % ('psf_flux',flux,flux_err)

    def _fit_match(self, dindex, sdata):
        if self.debug:
            print >>stderr,'\tfitting matched flux'
        niter=0
        ntry=0
        chi2per=PDEFVAL
        flux=DEFVAL
        flux_err=PDEFVAL
        bres={'flags':0,
              'flux':DEFVAL,'flux_err':PDEFVAL,
              'niter':0,'ntry':0,
              'chi2per':PDEFVAL, 'loglike':DEFVAL,
              'model':'nil'}


        if self.det_cat is None:
            # this is the detection band, just copy some data
            #flags,pars,pcov,niter0,ntry0,chi2per0,mod=\

            bres0=self._get_best_simple_pars(self.data,dindex)
            if bres0['flags']==0:
                bres.update(bres0)
                bres['flux']=bres['pars'][5]
                bres['flux_err']=sqrt(bres['pcov'][5,5])
                bres['model'] = bres0['model']

            else:
                bres['flags']=bres0['flags']
        else:
            bres0=self._get_best_simple_pars(self.det_cat,dindex)
            # if flags != 0 it is because we could not find a good fit of any
            # model
            if bres0['flags']==0:

                mod=bres0['model']
                bres['model'] = mod
                pars0=bres0['pars']

                print >>stderr,"    fitting: match flux (%s)" % mod

                if self.match_use_band_center:
                    pars0=self._set_center_from_band(dindex,pars0,mod)


                if False:
                    gm=gmix_image.gmix_fit.GMixFitMultiSimpleMatch(sdata['imlist'],
                                                                   sdata['wtlist'],
                                                                   sdata['jacob_list'],
                                                                   sdata['psf_gmix_list'],
                                                                   pars0,
                                                                   mod,
                                                                   lm_max_try=self.obj_ntry)
                else:
                    match_gmix = gmix_image.GMix(pars0, type=mod)
                    start_counts=self._get_match_start(dindex, mod, match_gmix)
                    match_gmix.set_psum(start_counts)

                    gm=GMixFitMultiMatch(sdata['imlist'],
                                         sdata['wtlist'],
                                         sdata['jacob_list'],
                                         sdata['psf_gmix_list'],
                                         match_gmix,
                                         lm_max_try=self.obj_ntry)

                res=gm.get_result()
                flags=res['flags']
                if flags==0:
                    print >>stderr,"        flux: %g match_flux: %g +/- %g" % (pars0[5],res['F'],res['flux_err'])
                    bres['flux']=res['F']
                    bres['flux_err']=res['flux_err']
                    bres['niter']=res['numiter']
                    bres['ntry']=res['ntry']
                    bres['chi2per']=res['chi2per']
                    bres['loglike'] = res['loglike']
                else:
                    bres['flags']=flags

            else:
                bres['flags']=bres0['flags']

        self.data['match_flags'][dindex] = bres['flags']
        self.data['match_model'][dindex] = bres['model']
        self.data['match_iter'][dindex] = bres['niter']
        self.data['match_tries'][dindex] = bres['ntry']
        self.data['match_chi2per'][dindex] = bres['chi2per']
        self.data['match_loglike'][dindex] = bres['loglike']
        self.data['match_flux'][dindex] = bres['flux']
        self.data['match_flux_err'][dindex] = bres['flux_err']
        if self.debug:
            fmt='\t\t%s[%s]: %g +/- %g'
            print >>stderr,fmt % ('match_flux',mod,bres['flux'],bres['flux_err'])

    def _set_center_from_band(self, dindex, pars0, mod):
        """
        For testing centroid stuff in match
        """
        pars=pars0.copy()
        print >>stderr,"trying band center instead of",pars[0:0+2]

        flagsn = '%s_flags' % mod
        if self.data[flagsn][dindex] == 0:
            parsn='%s_pars' % mod
            pars[0:0+2] = self.data[parsn][dindex,0:0+2]
            print >>stderr,"    using",mod,pars[0:0+2]
        elif self.data['psf_flags'][dindex] == 0:
            pars[0:0+2] = self.data['psf_pars'][dindex,0:0+2]
            print >>stderr,"    using psf",pars[0:0+2]
        else:
            print >>stderr,"    using det",pars[0:0+2]

        return pars

    def _get_match_start(self, dindex, mod, match_gmix):

        if mod=='exp':
            altmod='dev'
        else:
            altmod='exp'

        flagn = '%s_flags' % mod
        alt_flagn = '%s_flags' % altmod

        if self.data[flagn][dindex]==0:
            fn='%s_flux' % mod
            flux=self.data[fn][dindex]
        elif self.data[alt_flagn][dindex]==0:
            fn='%s_flux' % altmod
            flux=self.data[fn][dindex]
        elif self.data['psf_flags'][dindex]==0:
            flux=self.data['psf_flux'][dindex]
        else:
            flux=match_gmix.get_psum()

        return flux

    def _get_best_simple_pars(self, data, dindex):
        ddict={'flags':None,
               'pars':None,
               'pcov':None,
               'niter':0,
               'ntry':0,
               'chi2per':None,
               'loglike':None,
               'model':'nil'}

        flags=0
        nmod=len(self.simple_models)
        if nmod==2:

            expflags=data['exp_flags'][dindex]
            devflags=data['dev_flags'][dindex]

            if expflags==0 and devflags==0:
                if (data['exp_loglike'][dindex] 
                        > data['dev_loglike'][dindex]):
                    mod='exp'
                else:
                    mod='dev'
            elif expflags==0:
                mod='exp'
            elif devflags==0:
                mod='dev'
            else:
                flags |= (EXP_FIT_FAILURE+DEV_FIT_FAILURE)
                ddict['flags']=flags
                return ddict
        elif nmod==1:
            mod=self.simple_models[0]
            fn='%s_flags' % mod
            if data[fn][dindex] != 0:
                flags |= (EXP_FIT_FAILURE+DEV_FIT_FAILURE)
                ddict['flags']=flags
                return ddict
        else:
            raise ValueError("expected 1 or 2 simple models")

        pn='%s_pars' % mod
        pcn='%s_pars_cov' % mod
        itn='%s_iter' % mod
        tn='%s_tries' % mod
        chn='%s_chi2per' % mod
        ln='%s_loglike' % mod

        pars=data[pn][dindex].copy()
        pcov=data[pcn][dindex].copy()
        chi2per=data[chn][dindex]
        loglike=data[ln][dindex]

        if itn in data.dtype.names:
            niter=data[itn][dindex]
            ntry=data[tn][dindex]
        else:
            niter=int(DEFVAL)
            ntry=int(DEFVAL)

        return {'flags':flags,
                'pars':pars,
                'pcov':pcov,
                'niter':niter,
                'ntry':ntry,
                'chi2per':chi2per,
                'loglike':loglike,
                'model':mod}

    def _get_jacobian_list(self, meds, mindex):
        """
        Get a list of the jocobians for this object
        skipping the coadd
        """
        jacob_list=meds.get_jacobian_list(mindex)
        jacob_list=jacob_list[1:]
        return jacob_list

    def _get_psfex_reclist(self, meds, psfex_list, mindex):
        """
        Generate psfex reconstructions for the SE images
        associated with the cutouts, skipping the coadd

        add a little noise for the fitter
        """
        ncut=meds['ncutout'][mindex]
        imlist=[]
        ivarlist=[]
        cenlist=[]
        siglist=[]
        flist=[]
        for icut in xrange(1,ncut):
            file_id=meds['file_id'][mindex,icut]
            pex=psfex_list[file_id]
            fname=pex['filename']

            row=meds['orig_row'][mindex,icut]
            col=meds['orig_col'][mindex,icut]

            im0=pex.get_rec(row,col)
            cen0=pex.get_center(row,col)

            im,skysig=add_noise_matched(im0, PSF_S2N)
            ivar=1./skysig**2

            imlist.append( im )
            ivarlist.append(ivar)
            cenlist.append(cen0)
            siglist.append( pex.get_sigma() )
            flist.append( fname)

        return imlist, ivarlist, cenlist, siglist, flist, [row,col]



    def _get_all_psfex_objects(self, meds):
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

            pex=psfex.PSFEx(psfpath)
            psfex_list.append(pex)

        return psfex_list


    def _show(self, image):
        import images
        return images.multiview(image)


    def _print_simple_stats(self, ndict, rfc_res, res):                        
        fmt='\t\t%s: %g +/- %g'
        n=ndict
        if rfc_res['flags']==0:
            print >>stderr,'\t\trfc'
            nm='\t%s' % n['flux']
            flux=rfc_res['pars'][1]
            flux_err=sqrt(rfc_res['pcov'][1,1])
            print >>stderr,fmt % (nm,flux,flux_err)

            nm='\tT'
            flux=rfc_res['pars'][0]
            flux_err=sqrt(rfc_res['pcov'][0,0])
            print >>stderr, fmt % (nm,flux,flux_err)

        if res['flags']==0:
            nm=n['flux']
            flux=res['pars'][5]
            flux_err=sqrt(res['pcov'][5,5])
            print >>stderr,fmt % (nm,flux,flux_err)


    def _debug_image(self, im, wt):
        import biggles
        import images
        implt=images.multiview(im,show=False,title='image')
        wtplt=images.multiview(wt,show=False,title='weight')
        arr=biggles.Table(2,1)
        arr[0,0] = implt
        arr[1,0] = wtplt
        arr.show()
        key=raw_input('hit a key (q to quit): ')
        if key.lower() == 'q':
            stop

    def _set_index_list(self):
        """
        set the list of indices to be processed
        """
        if self.obj_range is None:
            start=0
            end=self.nobj-1
        else:
            start=self.obj_range[0]
            end=self.obj_range[1]

        self.index_list = numpy.arange(start,end+1)

    def _set_det_cat_and_struct(self, det_cat):
        # this creates self.data

        if self._checkpoint_data is not None:
            self.data=self._checkpoint_data
        else:
            self._make_struct()

        if det_cat is not None:
            if det_cat.size != self.meds.size:
                mess=("det_cat should be collated and match full "
                      "coadd size (%d) got %d")
                mess=mess % (self.meds.size,det_cat.size)
                raise ValueError(mess)
            self.det_cat=det_cat
        elif self.match_self:
            # do match flux on self!
            print >>stderr,"Will do match flux on self"
            self.det_cat = self.data
        else:
            self.det_cat=None


    def _write_checkpoint(self, tm):
        print >>stderr,'checkpointing at',tm,'seconds'
        print >>stderr,self.checkpoint_file
        fitsio.write(self.checkpoint_file,
                     self.data,
                     clobber=True)
        self.checkpointed=True

    def _make_struct(self):

        nobj=self.index_list.size

        dt=[('id','i4'),
            ('processed','i1'),
            ('flags','i4'),
            ('nimage_tot','i4'),
            ('nimage_use','i4'),
            ('time','f8')]

        simple_npars=6
        simple_models=self.simple_models
        for model in simple_models:
            n=get_model_names(model)

            dt+=[(n['rfc_flags'],'i4'),
                 (n['rfc_tries'],'i4'),
                 (n['rfc_iter'],'i4'),
                 (n['rfc_pars'],'f8',2),
                 (n['rfc_pars_cov'],'f8',(2,2)),
                 (n['flags'],'i4'),
                 (n['iter'],'i4'),
                 (n['tries'],'i4'),
                 (n['pars'],'f8',simple_npars),
                 (n['pars_cov'],'f8',(simple_npars,simple_npars)),
                 (n['flux'],'f8'),
                 (n['flux_err'],'f8'),
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

        dt += [('cmodel_flags','i4'),
               ('cmodel_iter','i4'),
               ('cmodel_tries','i4'),
               ('cmodel_flux','f8'),
               ('cmodel_flux_err','f8'),
               ('frac_dev','f8'),
               ('frac_dev_err','f8')]

        n=get_model_names('psf')
        dt += [('psf_flags','i4'),
               ('psf_iter','i4'),
               ('psf_tries','i4'),
               ('psf_pars','f8',3),
               ('psf_pars_cov','f8',(3,3)),
               ('psf_flux','f8'),
               ('psf_flux_err','f8'),
               ('psf_flux_s2n','f8'),
               (n['s2n_w'],'f8'),
               (n['loglike'],'f8'),
               (n['chi2per'],'f8'),
               (n['dof'],'f8'),
               (n['fit_prob'],'f8'),
               (n['aic'],'f8'),
               (n['bic'],'f8')]

        dt +=[('match_flags','i4'),
              ('match_tries','i4'),
              ('match_model','S3'),
              ('match_iter','i4'),
              ('match_chi2per','f8'),
              ('match_loglike','f8'),
              ('match_flux','f8'),
              ('match_flux_err','f8'),
              ]


        data=numpy.zeros(nobj, dtype=dt)
        #data['id'] = 1+self.index_list

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
        data['psf_flux_s2n'] = DEFVAL

        data['psf_s2n_w'] = DEFVAL
        data['psf_loglike'] = BIG_DEFVAL
        data['psf_chi2per'] = PDEFVAL
        data['psf_aic'] = BIG_PDEFVAL
        data['psf_bic'] = BIG_PDEFVAL


        data['match_flags'] = NO_ATTEMPT
        data['match_flux'] = DEFVAL
        data['match_flux_err'] = PDEFVAL
        data['match_chi2per'] = PDEFVAL
        data['match_loglike'] = DEFVAL
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

            data[n['s2n_w']] = DEFVAL
            data[n['loglike']] = BIG_DEFVAL
            data[n['chi2per']] = PDEFVAL
            data[n['aic']] = BIG_PDEFVAL
            data[n['bic']] = BIG_PDEFVAL
        
        self.data=data

    def _get_imlist_wtlist(self, meds, mindex):
        imlist,coadd=self._get_imlist(meds,mindex)
        wtlist=self._get_wtlist(meds,mindex)

        if self.reject_outliers:
            nreject=reject_outliers(imlist,wtlist)
            if self.make_plots:
                print 'nreject:',nreject
                plt=_show_used_pixels(imlist,wtlist,prompt=self.prompt)
                imname='mosaic%05d.png' % mindex
                print imname
                plt.write_img(1100,1100,imname)

        return imlist,wtlist,coadd

    def _get_imlist(self, meds, mindex, type='image'):
        """
        get the image list, skipping the coadd
        """
        imlist=meds.get_cutout_list(mindex,type=type)

        coadd=imlist[0].astype('f8')
        imlist=imlist[1:]

        imlist = [im.astype('f8') for im in imlist]
        return imlist, coadd


    def _get_wtlist(self, meds, mindex):
        """
        get the weight list.

        If using the seg map, mark pixels outside the coadd object region as
        zero weight
        """
        if self.region=='seg_and_sky':
            wtlist=meds.get_cweight_cutout_list(mindex)
            wtlist=wtlist[1:]

            wtlist=[wt.astype('f8') for wt in wtlist]
        else:
            raise ValueError("support other region types")
        return wtlist

    def _show_coadd(self):
        import images
        images.view(self.coadd)




_stat_names=['s2n_w',
             'loglike',
             'chi2per',
             'dof',
             'fit_prob',
             'aic',
             'bic']

def get_model_names(model):
    names=['rfc_flags',
           'rfc_tries',
           'rfc_iter',
           'rfc_pars',
           'rfc_pars_cov',
           'flags',
           'pars',
           'pars_cov',
           'logpars',
           'logpars_cov',
           'flux',
           'flux_err',
           'flux_cov',
           'gmix_pars',
           'g',
           'g_cov',
           'g_sens',
           'e',
           'e_cov',
           'e_sens',
           'P',
           'Q',
           'R',
           'iter',
           'tries',
           'arate',
           'tau']
    names += _stat_names

    ndict={}
    for n in names:
        ndict[n] = '%s_%s' % (model,n)

    return ndict


def add_noise_matched(im, s2n):
    """
    Add gaussian noise to an image assuming a matched filter is used.

     sum(pix^2)
    ------------ = S/N^2
      skysig^2

    thus
        
    sum(pix^2)
    ---------- = skysig^2
      (S/N)^2

    parameters
    ----------
    im: numpy array
        The image
    s2n:
        The requested S/N

    outputs
    -------
    image, skysig
        A tuple with the image and error per pixel.

    """
    from math import sqrt

    skysig2 = (im**2).sum()/s2n**2
    skysig = sqrt(skysig2)

    noise_image = skysig*randn(im.size).reshape(im.shape)
    image = im + noise_image

    return image, skysig


def sigma_clip(arrin, niter=4, nsig=4, get_indices=False, extra={}, 
               verbose=False, silent=False):
    """
    NAME:
      sigma_clip()
      
    PURPOSE:
      Calculate the mean/stdev of an array with sigma clipping. Iterate
      niter times, removing elements that are outside nsig, and recalculating
      mean/stdev.

    CALLING SEQUENCE:
      mean,stdev = sigma_clip(arr, niter=4, nsig=4, extra={})
    
    INPUTS:
      arr: A numpy array or a sequence that can be converted.

    OPTIONAL INPUTS:
      niter: number of iterations, defaults to 4
      nsig: number of sigma, defaults to 4
      get_indices: bool,optional
        if True return mean,stdev,indices

    OUTPUTS:
      mean,stdev: A tuple containing mean and standard deviation.
    OPTIONAL OUTPUTS
      extra={}: Dictionary containing the array of used indices in
         extra['index']

    REVISION HISTORY:
      Converted from IDL: 2006-10-23. Erin Sheldon, NYU
      Minor bug fix to error messaging: 2010-05-28. Brian Gerke, SLAC
      Added silent keyword, to shut off error messages.  BFG 2010-09-13

    """
    arr = numpy.array(arrin, ndmin=1, copy=False)

    index = numpy.arange( arr.size )

    if get_indices:
        res=[None,None,None]
    else:
        res=[None,None]

    for i in numpy.arange(niter):
        m = arr[index].mean()
        s = arr[index].std()

        if verbose:
            stdout.write('iter %s\tnuse: %s\tmean %s\tstdev %s\n' % \
                             (i+1, index.size,m,s))

        clip = nsig*s

        w, = numpy.where( (numpy.abs(arr[index] - m)) < clip )

        if (w.size == 0):
            if (not silent):
                stderr.write("nsig too small. Everything clipped on "
                             "iteration %d\n" % (i+1))
            res[0]=m
            res[1]=s
            return res

        index = index[w]

    # Calculate final stats
    amean = arr[index].mean()
    asig = arr[index].std()

    res[0]=m
    res[1]=s
    extra['index'] = index
    if get_indices:
        res[2] = index

    return res 


def reroot_path(psfpath, old_desdata):
    desdata=os.environ['DESDATA']


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

def _show_used_pixels(imlist0,wtlist, prompt=True):
    import images
    oplt=images.view_mosaic(imlist0,show=False)
    oplt.title='original'

    imlist=[]
    for i in xrange(len(imlist0)):
        im0=imlist0[i]
        wt=wtlist[i]

        im=wt*im0

        imlist.append(im)
    
    plt=images.view_mosaic(imlist,show=False)
    plt.title='image*weight'

    if prompt:
        oplt.show()
        plt.show()
        key=raw_input('hit a key (q to quit): ')
        if key.lower()=='q':
            stop
        
    return plt
