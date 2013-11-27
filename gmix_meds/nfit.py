"""
todo

    - make sure all metadata is being copied
    - implement exp g prior
    - figure out what to do with T and counts priors
    - scale images by jacobian determinant?
    - copy psf data

"""
import os
from sys import stderr,stdout
import time
import numpy
import meds
import psfex
import ngmix
from ngmix import srandu
from ngmix import GMixMaxIterEM, print_pars

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

NO_ATTEMPT=2**30

#PSF_S2N=1.e6
PSF_OFFSET_MAX=0.25
PSF_TOL=1.0e-5
EM_MAX_TRY=3
EM_MAX_ITER=100

SIMPLE_MODELS_DEFAULT = ['exp','dev']


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
        psf_model: string, int
            e.g. "em2"
        psf_offset_max: optional
            max offset between multi-component gaussians in psf models

        checkpoint: number, optional
            Time after which to checkpoint, seconds
        checkpoint_file: string, optional
            File which will hold a checkpoint.
        checkpoint_data: dict, optional
            The data representing a previous checkpoint, object and
            psf fits
        """

        self.conf={}
        self.conf.update(keys)

        self.imstart=1
        self.fit_types=self.conf['fit_types']
        self.simple_models=keys.get('simple_models',SIMPLE_MODELS_DEFAULT )

        self.guess_from_coadd=keys.get('guess_from_coadd',False)

        self.nwalkers=keys.get('nwalkers',20)
        self.burnin=keys.get('burnin',400)
        self.nstep=keys.get('nstep',200)
        self.do_pqr=keys.get("do_pqr",False)
        self.do_lensfit=keys.get("do_lensfit",False)
        self.mca_a=keys.get('mca_a',2.0)

        self._unpack_priors()

        self.meds_files=_get_as_list(meds_files)
        self.checkpoint = self.conf.get('checkpoint',172800)
        self.checkpoint_file = self.conf.get('checkpoint_file',None)
        self._set_checkpoint_data(**keys)

        self.nband=len(self.meds_files)
        self.iband = range(self.nband)

        self._load_meds_files()
        self.psfex_lol = self._get_psfex_lol()

        self.obj_range=keys.get('obj_range',None)
        self._set_index_list()

        self.psf_model=keys.get('psf_model','em2')
        self.psf_offset_max=keys.get('psf_offset_max',PSF_OFFSET_MAX)
        self.psf_ngauss=get_psf_ngauss(self.psf_model)

        self.debug=keys.get('debug',0)

        self.psf_ntry=keys.get('psf_ntry', EM_MAX_TRY)
        self.psf_maxiter=keys.get('psf_maxiter', EM_MAX_ITER)
        self.psf_tol=keys.get('psf_tol', PSF_TOL)

        self.region=keys.get('region','seg_and_sky')
        self.max_box_size=keys.get('max_box_size',2048)

        self.reject_outliers=keys.get('reject_outliers',False) # from cutouts

        self.make_plots=keys.get('make_plots',False)
        self.prompt=keys.get('prompt',True)

        if self._checkpoint_data is None:
            self._make_struct()
            self._make_psf_struct()

    def _unpack_priors(self):
        conf=self.conf

        nmod=len(self.simple_models)
        T_priors=conf['T_priors']
        g_priors=conf['g_priors']

        if (len(T_priors) != nmod or len(g_priors) != nmod ):
            raise ValueError("models and T,g priors must be same length")

        priors={}
        models=self.simple_models
        for i in xrange(nmod):
            model=models[i]
            T_prior=T_priors[i]
            g_prior=g_priors[i]
            
            modlist={'T':T_prior, 'g':g_prior}
            priors[model] = modlist

        self.priors=priors
        self.draw_g_prior=conf.get('draw_g_prior',True)

        # in arcsec (or units of jacobian)
        self.cen_prior=conf.get("cen_prior",None)

    def _set_checkpoint_data(self, **keys):
        self._checkpoint_data=keys.get('checkpoint_data',None)
        if self._checkpoint_data is not None:
            self.data=self._checkpoint_data['data']
            self.psf_data=self._checkpoint_data['psf_data']

    def get_data(self):
        """
        Get the data structure.  If a subset was requested, only those rows are
        returned.
        """
        return self.data

    def get_psf_data(self):
        """
        Get the psf data structure.
        """
        return self.psf_data

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
        self.checkpointed=False

        for dindex in xrange(num):
            if self.data['processed'][dindex]==1:
                # checkpointing
                continue

            mindex = self.index_list[dindex]
            print >>stderr,'index: %d:%d' % (mindex,last),
            self.fit_obj(dindex)

            tm=time.time()-t0

            if self._should_checkpoint(tm):
                self._write_checkpoint(tm)

        tm=time.time()-t0
        print >>stderr,"time:",tm
        print >>stderr,"time per:",tm/num


    def fit_obj(self, dindex):
        """
        Process the indicated object through the requested fits
        """

        t0=time.time()

        # for checkpointing
        self.data['processed'][dindex]=1

        mindex = self.index_list[dindex]

        # need to do this because we work on subset files
        self.data['id'][dindex] = self.meds_list[0]['number'][mindex]

        self.data['flags'][dindex] = self._obj_check(mindex)
        if self.data['flags'][dindex] != 0:
            return 0

        # lists of lists.  Coadd lists are length 1
        im_lol,wt_lol,coadd_im_lol,coadd_wt_lol = \
                self._get_imlol_wtlol(dindex,mindex)
        jacob_lol,coadd_jacob_lol=self._get_jacobian_lol(mindex)

        print >>stderr,im_lol[0][0].shape
    
        print >>stderr,'    fitting: coadd psf models'
        coadd_keep_lol,coadd_psf_gmix_lol,coadd_flags=\
                self._fit_psfs(dindex,coadd_jacob_lol,do_coadd=True)
        if any(coadd_flags):
            self.data['flags'][dindex] = PSF_FIT_FAILURE 
            return

        print >>stderr,'    fitting: psf models'
        keep_lol,psf_gmix_lol,flags=self._fit_psfs(dindex,jacob_lol)
        if any(flags):
            self.data['flags'][dindex] = PSF_FIT_FAILURE 
            return

        coadd_im_lol, coadd_wt_lol, coadd_jacob_lol, coadd_len_list = \
            self._extract_sub_lists(coadd_keep_lol,coadd_im_lol,coadd_wt_lol,coadd_jacob_lol)
        if any([l==0 for l in coadd_len_list]):
            print >>stderr,'wierd coadd lists zero length!'
            self.data['flags'][dindex] = PSF_FIT_FAILURE 
            return

        im_lol, wt_lol, jacob_lol, len_list = \
            self._extract_sub_lists(keep_lol,im_lol,wt_lol,jacob_lol)
        if any([l==0 for l in len_list]):
            print >>stderr,'wierd lists zero length!'
            self.data['flags'][dindex] = PSF_FIT_FAILURE 
            return

        self.data['nimage_use'][dindex, :] = len_list

        sdata={'keep_lol':keep_lol,
               'im_lol':im_lol,
               'wt_lol':wt_lol,
               'jacob_lol':jacob_lol,
               'psf_gmix_lol':psf_gmix_lol,

               'coadd_keep_lol':coadd_keep_lol,
               'coadd_im_lol':coadd_im_lol,
               'coadd_wt_lol':coadd_wt_lol,
               'coadd_jacob_lol':coadd_jacob_lol,
               'coadd_psf_gmix_lol':coadd_psf_gmix_lol}

        self._fit_all_models(dindex, sdata)

        self.data['time'][dindex] = time.time()-t0


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
            print >>stderr,'Box size too big:',box_size
            flags |= BOX_SIZE_TOO_BIG

        if meds['ncutout'][mindex] < 2:
            print >>stderr,'No SE cutouts'
            flags |= NO_SE_CUTOUTS
        return flags


    def _get_imlol_wtlol(self, dindex, mindex):
        """
        Get a list of the jocobians for this object
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
                    print >>stderr,'        rejected:',nreject

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
        Get a list of the jocobians for this object
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
        Get a list of the jocobians for this object
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
        """
        ptuple = self._get_psfex_reclist(meds, psfex_list, dindex,do_coadd=do_coadd)
        imlist,cenlist,siglist,flist=ptuple

        psf_start = self.data['psf_start'][dindex, band]

        keep_list=[]
        gmix_list=[]

        flags=0

        gmix_psf=None
        for i in xrange(len(imlist)):

            psf_index = psf_start + i

            im=imlist[i]
            jacob0=jacob_list[i]
            sigma=siglist[i]

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
                self._set_psf_data(psf_index, gmix_psf)

                keep,offset_arcsec=self._should_keep_psf(gmix_psf)
                if keep:
                    gmix_list.append( gmix_psf )
                    keep_list.append(i)
                else:
                    print >>stderr,('large psf offset: %s '
                                    'in %s' % (offset_arcsec,flist[i]))
                    tflags |= PSF_LARGE_OFFSETS 

                
            except GMixMaxIterEM:
                print >>stderr,'psf fail',flist[i]

                tlags == PSF_FIT_FAILURE

            self.psf_data['flags'][psf_index] = tflags
            flags |= tflags

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

        return imlist, cenlist, siglist, flist

    def _should_keep_psf(self, gm):
        """
        For double gauss we limit the separation
        """
        keep=True
        if self.psf_ngauss == 2:
            offset_arcsec = calc_offset_arcsec(gm)
            if offset_arcsec > self.psf_offset_max:
                keep=False

        return keep, offset_arcsec

    def _do_fit_psf(self, im, jacob, sigma_guess, first_guess=None):
        """
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
                print >>stderr,'last fit:'
                print >>stderr,fitter.get_gmix()
                print >>stderr,'try:',i+1,'fdiff:',res['fdiff'],'numiter:',res['numiter']
                if i == (self.psf_ntry-1):
                    raise

        return fitter

    def _get_em_guess(self, sigma2):
        """
        Guess for the EM algorithm
        """
        if self.psf_ngauss==1:
            pars=numpy.array( [1.0, 0.0, 0.0, 
                               sigma2*(1.0 + 0.1*srandu()),
                               0.0,
                               sigma2*(1.0 + 0.1*srandu())] )
        else:

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

        return ngmix.gmix.GMix(pars=pars)




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


    def _fit_all_models(self, dindex, sdata):
        """
        Fit psf flux and other models
        """
        self._fit_psf_flux(dindex, sdata)
 
        s2n=self.data['psf_flux'][dindex,:]/self.data['psf_flux_err'][dindex,:]
        max_psf_s2n=numpy.nanmax(s2n)
         
        if max_psf_s2n >= self.conf['min_psf_s2n']:
            if 'simple' in self.fit_types:
                self._fit_simple_models(dindex, sdata)
        else:
            mess="    psf s/n too low: %s (%s)"
            mess=mess % (max_psf_s2n,self.conf['min_psf_s2n'])
            print >>stderr,mess


    def _fit_psf_flux(self, dindex, sdata):
        """
        Perform PSF flux fits on each band separately
        """

        print >>stderr,'    fitting: psf'
        for band in self.iband:
            self._fit_psf_flux_oneband(dindex, sdata, band)

        print_pars(self.data['psf_flux'][dindex],     stream=stderr, front='        ')
        print_pars(self.data['psf_flux_err'][dindex], stream=stderr, front='        ')


    def _fit_psf_flux_oneband(self, dindex, sdata, band):
        """
        Fit the PSF flux in a single band
        """
        fitter=ngmix.fitting.PSFFluxFitter(sdata['im_lol'][band],
                                           sdata['wt_lol'][band],
                                           sdata['jacob_lol'][band],
                                           sdata['psf_gmix_lol'][band])
        fitter.go()
        res=fitter.get_result()
        self.data['psf_flags'][dindex,band] = res['flags']
        self.data['psf_flux'][dindex,band] = res['flux']
        self.data['psf_flux_err'][dindex,band] = res['flux_err']
        self.data['psf_chi2per'][dindex,band] = res['chi2per']
        self.data['psf_dof'][dindex,band] = res['dof']

    def _fit_simple_models(self, dindex, sdata):
        """
        Fit all the simple models
        """

        for model in self.simple_models:
            print >>stderr,'    fitting:',model

            gm=self._fit_simple(dindex, model, sdata)

            res=gm.get_result()

            self._copy_simple_pars(dindex, res)
            self._print_simple_res(res)

            if self.make_plots:
                mindex = self.index_list[dindex]
                ptrials,presid_list=gm.make_plots(title='%s multi-epoch' % model,
                                                  do_residual=True)
                ptrials.write_img(1200,1200,'trials-%06d-%s.png' % (mindex,model))
                if presid_list is not None:
                    for band,plt in enumerate(presid_list):
                        plt.write_img(1920,1200,'resid-%06d-%s-band%d.png' % (mindex,model,band))

    def _fit_simple(self, dindex, model, sdata):
        """
        Fit one of the "simple" models, e.g. exp or dev
        """

        if self.guess_from_coadd:
            full_guess=self._get_full_simple_guess(dindex, model, sdata)
            counts_guess=None
            T_guess=None
        else:
            full_guess=None
            counts_guess=self._get_counts_guess(dindex,sdata)
            T_guess=self._get_T_guess(dindex,sdata)
        
        gm=self._do_fit_simple(model, 
                               sdata['im_lol'],
                               sdata['wt_lol'],
                               sdata['jacob_lol'],
                               sdata['psf_gmix_lol'],
                               self.burnin,
                               self.nstep,
                               T_guess=T_guess,
                               counts_guess=counts_guess,
                               full_guess=full_guess)
        return gm


    def _get_counts_guess(self, dindex, sdata):
        """
        Based on the psf flux guess
        """
        psf_flux=self.data['psf_flux'][dindex,:].clip(min=0.1, max=None)
        return psf_flux

    def _get_T_guess(self, dindex, sdata):
        """
        Guess at T in arcsec**2

        Guess corresponds to FWHM=2.0 arcsec

        Assuming scale is 0.27''/pixel
        """
        return 1.44

    def _get_full_simple_guess(self, dindex, model, sdata):
        print >>stderr,'    getting guess from coadd'
        counts_guess=self._get_counts_guess(dindex,sdata)
        T_guess=self._get_T_guess(dindex,sdata)

        gm=self._do_fit_simple(model,
                               sdata['coadd_im_lol'],
                               sdata['coadd_wt_lol'],
                               sdata['coadd_jacob_lol'],
                               sdata['coadd_psf_gmix_lol'],
                               self.conf['guess_burnin'],
                               self.conf['guess_nstep'],
                               T_guess=T_guess,
                               counts_guess=counts_guess)

        res=gm.get_result()
        self._print_simple_res(res)

        if self.make_plots:
            mindex = self.index_list[dindex]
            ptrials,presid_list=gm.make_plots(title='%s coadd' % model,
                                              do_residual=True)
            ptrials.write_img(1200,1200,'trials-%06d-%s-coadd.png' % (mindex,model))
            if presid_list is not None:
                for band,plt in enumerate(presid_list):
                    plt.write_img(1920,1200,'resid-%06d-%s-band%d-coadd.png' % (mindex,model,band))

        return gm.get_trials()

    def _do_fit_simple(self, model, im_lol, wt_lol, jacob_lol, psf_gmix_lol,
                       burnin,nstep,
                       T_guess=None,
                       counts_guess=None,
                       full_guess=None):

        priors=self.priors[model]
        g_prior=priors['g']
        T_prior=priors['T']

        gm=ngmix.fitting.MCMCSimple(im_lol,
                                    wt_lol,
                                    jacob_lol,
                                    model,
                                    psf=psf_gmix_lol,

                                    nwalkers=self.nwalkers,
                                    burnin=burnin,
                                    nstep=nstep,
                                    mca_a=self.mca_a,

                                    iter=True,

                                    T_guess=T_guess,
                                    counts_guess=counts_guess,

                                    full_guess=full_guess,

                                    cen_prior=self.cen_prior,
                                    T_prior=T_prior,
                                    g_prior=g_prior,
                                    draw_g_prior=self.draw_g_prior,
                                    do_lensfit=self.do_lensfit,
                                    do_pqr=self.do_pqr)
        gm.go()
        return gm


    def _copy_simple_pars(self, dindex, res):
        """
        Copy from the result dict to the output array
        """
        model=res['model']
        n=get_model_names(model)

        self.data[n['flags']][dindex] = res['flags']

        if res['flags'] == 0:
            pars=res['pars']
            pars_cov=res['pars_cov']

            flux=pars[5:]
            flux_cov=pars_cov[5:, 5:]

            self.data[n['pars']][dindex,:] = pars
            self.data[n['pars_cov']][dindex,:,:] = pars_cov

            self.data[n['flux']][dindex] = flux
            self.data[n['flux_cov']][dindex] = flux_cov

            self.data[n['g']][dindex,:] = res['g']
            self.data[n['g_cov']][dindex,:,:] = res['g_cov']

            self.data[n['arate']][dindex] = res['arate']
            if res['tau'] is not None:
                self.data[n['tau']][dindex] = res['tau']

            for sn in _stat_names:
                self.data[n[sn]][dindex] = res[sn]

            if self.do_lensfit:
                self.data[n['g_sens']][dindex,:] = res['g_sens']

            if self.do_pqr:
                self.data[n['P']][dindex] = res['P']
                self.data[n['Q']][dindex,:] = res['Q']
                self.data[n['R']][dindex,:,:] = res['R']
                


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

        self.nobj_tot = self.meds_list[0].size

    def _get_psfex_lol(self):
        """
        Load psfex objects for each of the SE images
        include the coadd so we get  the index right
        """
        print >>stderr,'loading psfex'
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

    def _should_checkpoint(self, tm):
        """
        Should we write a checkpoint file?
        """
        if (tm > self.checkpoint
                and self.checkpoint_file is not None
                and not self.checkpointed):
            return True
        else:
            return False

    def _print_simple_res(self, res):
        if res['flags']==0:
            self._print_simple_fluxes(res)
            self._print_simple_T(res)
            self._print_simple_shape(res)
            print >>stderr,'        arate:',res['arate']

    def _print_simple_shape(self, res):
        g1=res['pars'][2]
        g1err=numpy.sqrt(res['pars_cov'][2,2])
        g2=res['pars'][3]
        g2err=numpy.sqrt(res['pars_cov'][3,3])

        print >>stderr,'        g1: %.4g +/- %.4g g2: %.4g +/- %.4g' % (g1,g1err,g2,g2err)

    def _print_simple_fluxes(self, res):
        """
        print in a nice format
        """
        from numpy import sqrt,diag
        flux=res['pars'][5:]
        flux_cov=res['pars_cov'][5:, 5:]
        flux_err=sqrt(diag(flux_cov))

        print_pars(flux,     stream=stderr, front='        ')
        print_pars(flux_err, stream=stderr, front='        ')

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
        print >>stderr, '        T: %s +/- %s Ts2n: %s sigma: %s' % tup

    def _write_checkpoint(self, tm):
        import fitsio
        print >>stderr,'checkpointing at',tm,'seconds'
        print >>stderr,self.checkpoint_file
        with fitsio.FITS(self.checkpoint_file,'rw',clobber=True) as fobj:
            fobj.write(self.data, extname="model_fits")
            fobj.write(self.psf_data, extname="psf_fits")
        self.checkpointed=True

    def _count_all_cutouts(self):
        """
        Count the cutouts for the objects, not including the coadd cutouts.  If
        obj_range was sent, this will be a subset
        """
        ncutout=0
        ncoadd=self.index_list.size
        for meds in self.meds_list:
            ncutout += meds['ncutout'][self.index_list].sum() - ncoadd
        return ncutout


    def _set_psf_data(self, index, gm):
        """
        Set psf fit data. Index can be got from the main model
        fits struct
        """
        pars=gm.get_full_pars()
        g1,g2,T=gm.get_g1g2T()
        self.psf_data['pars'][index,:] = pars
        self.psf_data['g'][index,0] = g1
        self.psf_data['g'][index,1] = g2
        self.psf_data['T'][index] = T


    def _make_psf_struct(self):
        """
        We will make the maximum number of possible psfs according
        to the cutout count, not counting the coadd
        """

        npars=self.psf_ngauss*6
        dt=[('id','i4'), # same as 'id' in main struct, used for matching
            ('band','i2'),
            ('file_id','i2'), # to determine the psf file
            ('flags','i4'),
            ('g','f8',2),
            ('T','f8'),
            ('pars','f8',npars)]

        ncutout=self._count_all_cutouts()
        if ncutout > 0:
            psf_data = numpy.zeros(ncutout, dtype=dt)

            psf_data['g'] = PDEFVAL
            psf_data['T'] = PDEFVAL
            psf_data['pars'] = PDEFVAL
            psf_data['flags'] = NO_ATTEMPT

            self._set_psf_start()
        else:
            psf_data=numpy.zeros(1)
        self.psf_data=psf_data

    def _set_psf_start(self):
        """
        Set the psf start in the self.data struct and info and fill in some psf
        metadata in the self.psf_data struct
        """
        print >>stderr,'Setting psf start positions'
        n=self.data.size

        data=self.data
        psf_data=self.psf_data

        data['psf_start'] = -1

        beg=0
        for dindex in xrange(n):

            mindex=self.index_list[dindex]

            for band,meds in enumerate(self.meds_list):
                # minus one to remove coadd
                ncut_tot = meds['ncutout'][mindex]
                ncut_se  = ncut_tot-1
                if ncut_se > 1:
                    end=beg+ncut_se
                    data['psf_start'][dindex, band] = beg

                    psf_data['band'][beg:end] = band 
                    psf_data['id'][beg:end] = dindex
                    psf_data['file_id'][beg:end] = meds['file_id'][mindex,1:ncut_tot]

                    beg += ncut_se


    def _make_struct(self):
        """
        make the output structure
        """
        nband=self.nband
        bshape=(nband,)

        dt=[('id','i4'),
            ('processed','i1'),
            ('flags','i4'),
            ('nimage_tot','i4',bshape),
            ('nimage_use','i4',bshape),
            ('psf_start','i4',bshape),  # pointers into psf file
            ('time','f8')]

        # the psf fits are done for each band separately
        n=get_model_names('psf')
        dt += [(n['flags'],   'i4',bshape),
               (n['flux'],    'f8',bshape),
               (n['flux_err'],'f8',bshape),
               (n['chi2per'],'f8',bshape),
               (n['dof'],'f8',bshape)]
       
        if 'simple' in self.fit_types:
    
            simple_npars=5+nband
            simple_models=self.simple_models

            if nband==1:
                cov_shape=(nband,)
            else:
                cov_shape=(nband,nband)

            for model in simple_models:
                n=get_model_names(model)

                np=simple_npars


                dt+=[(n['flags'],'i4'),
                     (n['pars'],'f8',np),
                     (n['pars_cov'],'f8',(np,np)),
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
                if self.do_lensfit:
                    dt += [(n['g_sens'], 'f8', 2)]
                if self.do_pqr:
                    dt += [(n['P'], 'f8'),
                           (n['Q'], 'f8', 2),
                           (n['R'], 'f8', (2,2))]


        num=self.index_list.size
        data=numpy.zeros(num, dtype=dt)

        data['psf_flags'] = NO_ATTEMPT
        data['psf_flux'] = DEFVAL
        data['psf_flux_err'] = PDEFVAL

        if 'simple' in self.fit_types:
            for model in simple_models:
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

                if self.do_lensfit:
                    data[n['g_sens']] = DEFVAL
                if self.do_pqr:
                    data[n['P']] = DEFVAL
                    data[n['Q']] = DEFVAL
                    data[n['R']] = DEFVAL

     
        self.data=data

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


