from sys import stderr
import time
import numpy
from numpy import sqrt
from numpy.random import randn
import fitsio
import meds
import gmix_image
from gmix_image.gmix_fit import LM_MAX_TRY, \
        GMixFitPSFJacob,\
        GMixFitMultiSimple,GMixFitMultiCModel, \
        GMixFitMultiPSFFlux,GMixFitMultiMatch
from gmix_image.gmix_em import GMixEMBoot
import psfex

DEFVAL=-9999
PDEFVAL=9999
BIG_DEFVAL=-9.999e9
BIG_PDEFVAL=9.999e9


NO_SE_CUTOUTS=2**0
PSF_FIT_FAILURE=2**1
EXP_FIT_FAILURE=2**2
DEV_FIT_FAILURE=2**3

NO_ATTEMPT=2**30

PSF_S2N=1.e6

_psf_ngauss_map={'lm1':1, 'lm2':2, 'lm3':3,
                 'em1':1, 'em2':2, 'em3':3}
def get_psf_ngauss(psf_model):
    if psf_model not in _psf_ngauss_map:
        raise ValueError("bad psf model: '%s'" % psf_model)
    return _psf_ngauss_map[psf_model]

class MedsFit(object):
    def __init__(self,
                 meds_file,
                 seed=None,
                 obj_range=None,
                 det_cat=None,
                 psf_model="lm2",
                 psf_ntry=LM_MAX_TRY,
                 obj_ntry=2,
                 reject_outliers=False,
                 pix_nsig=10,
                 debug=0):
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
        
        numpy.random.seed(seed)

        self.meds_file=meds_file
        self.meds=meds.MEDS(meds_file)
        self.meds_meta=self.meds.get_meta()
        self.nobj=self.meds.size

        self.obj_range=obj_range

        self.psf_model=psf_model
        self.psf_ngauss=get_psf_ngauss(psf_model)

        self.debug=debug

        self.psf_ntry=psf_ntry
        self.obj_ntry=obj_ntry
        self.reject_outliers=reject_outliers
        self.pix_nsig=pix_nsig

        self.simple_models=['exp','dev']

        self._set_index_list()
        self._make_struct()
        self._set_det_cat(det_cat)
        self._load_all_psfex_objects()

    def get_data(self):
        """
        Get the data structure.  If a subset was requested, only those rows are
        returned.
        """
        return self.data[self.index_list]

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
        last=self.index_list[-1]
        for index in self.index_list:
            print >>stderr,'index: %d:%d' % (index,last)
            self.fit_obj(index)

    def fit_obj(self, index):
        """
        Process the indicated object

        The first cutout is always the coadd, followed by
        the SE images which will be fit simultaneously
        """
        if self.meds['ncutout'][index] < 2:
            self.data['flags'][index] |= NO_SE_CUTOUTS
            return

        t0=time.time()
        if self.reject_outliers:
            imlist,wtlist=self._get_imlists_outlier_reject(index)
        else:
            imlist=self._get_imlist(index)
            wtlist=self._get_wtlist(index)

        jacob_list=self._get_jacobian_list(index)

        self.data['nimage'][index] = len(imlist)

        if self.debug: print >>stderr,'\tfitting psfs'

        psf_gmix_list=self._fit_psfs(index,jacob_list)
        if psf_gmix_list is None:
            self.data['flags'][index] |= PSF_FIT_FAILURE
            self.data['time'][index] = time.time()-t0
            return

        sdata={'imlist':imlist,'wtlist':wtlist,
               'jacob_list':jacob_list,
               'psf_gmix_list':psf_gmix_list}

        self._fit_simple_models(index, sdata)
        self._fit_cmodel(index, sdata)
        self._fit_psf_flux(index, sdata)
        # might just be a copy if there is not det_cat
        self._fit_match(index, sdata)

        if self.debug >= 3:
            self._debug_image(sdata['imlist'][0],sdata['wtlist'][-1])

        self.data['time'][index] = time.time()-t0

    def _fit_psfs(self,index,jacob_list):
        """
        Generate psfex images for all SE images and fit
        them to gaussian mixture models
        """
        ptuple = self._get_psfex_reclist(index)
        imlist,ivarlist,cenlist,siglist,flist,cenpix=ptuple

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
            if res['flags'] != 0:
                print >>stderr,'psf fitting failed, '
                return None


            gmix_psf=gm.get_gmix()
            gmix_list.append( gmix_psf )

            if False:
                self._compare_psf_model(im, gm, index, i, flist[i], cenpix)

        return gmix_list


    def _do_fit_psf(self, im, jacob, ivar, sigma_guess):
        if 'lm' in self.psf_model:
            gm=self._run_psf_lm_fit(im,ivar,jacob)
        elif 'em' in self.psf_model:
            gm=self._run_psf_em_fit(im,ivar,jacob,sigma_guess)
        else:
            raise RuntimeError("bad psf model '%s'" % self.psf_model)
        return gm

    def _run_psf_lm_fit(self, im, ivar, jacob):
        gm=GMixFitPSFJacob(im,
                           ivar,
                           jacob,
                           self.psf_ngauss,
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


    def _compare_psf_model(self, im, gm, index, i,fname,cenpix):
        """
        Since we work in sky coords, can only generate the
        diff image currently
        """
        import os
        import images

        print fname
        name='%s_%06d_%02d' % (self.psf_model,index,i)

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
            rfc_res=gm.get_rfc_result()

            n=get_model_names(model)

            if self.debug:
                self._print_simple_stats(n, rfc_res, res)

            self._copy_simple_pars(index, rfc_res, res, n )

    def _copy_simple_pars(self, index, rfc_res, res, n):
        self.data[n['rfc_flags']][index] = rfc_res['flags']
        self.data[n['rfc_iter']][index] = rfc_res['numiter']
        self.data[n['rfc_tries']][index] = rfc_res['ntry']

        if rfc_res['flags']==0:
            self.data[n['rfc_pars']][index,:] = rfc_res['pars']
            self.data[n['rfc_pars_cov']][index,:] = rfc_res['pcov']

        self.data[n['flags']][index] = res['flags']
        self.data[n['iter']][index] = res['numiter']
        self.data[n['tries']][index] = res['ntry']

        if res['flags'] == 0:
            self.data[n['pars']][index,:] = res['pars']
            self.data[n['pars_cov']][index,:,:] = res['pcov']

            flux=res['pars'][5]
            flux_err=sqrt(res['pcov'][5,5])
            self.data[n['flux']][index] = flux
            self.data[n['flux_err']][index] = flux_err

            self.data[n['g']][index,:] = res['pars'][2:2+2]
            self.data[n['g_cov']][index,:,:] = res['pcov'][2:2+2,2:2+2]

            for sn in _stat_names:
                self.data[n[sn]][index] = res[sn]
        else:
            if self.debug:
                print >>stderr,'flags != 0, errmsg:',res['errmsg']
            if self.debug > 1 and self.debug < 3:
                self._debug_image(sdata['imlist'][0],sdata['wtlist'][0])



    def _fit_simple(self, model, sdata):
        """
        Fit one of the "simple" models, e.g. exp or dev
        """
        gm=GMixFitMultiSimple(sdata['imlist'],
                              sdata['wtlist'],
                              sdata['jacob_list'],
                              sdata['psf_gmix_list'],
                              model,
                              lm_max_try=self.obj_ntry)
        return gm

    def _fit_cmodel(self, index, sdata):
        if self.debug:
            print >>stderr,'\tfitting frac_dev'
        if self.data['exp_flags'][index]!=0:
            self.data['cmodel_flags'][index] |= EXP_FIT_FAILURE
        if self.data['dev_flags'][index]!=0:
            self.data['cmodel_flags'][index] |= DEV_FIT_FAILURE

        if self.data['cmodel_flags'][index] != 0:
            return

        exp_gmix = gmix_image.GMix(self.data['exp_pars'][index],type='exp')
        dev_gmix = gmix_image.GMix(self.data['dev_pars'][index],type='dev')

        gm=GMixFitMultiCModel(sdata['imlist'],
                              sdata['wtlist'],
                              sdata['jacob_list'],
                              sdata['psf_gmix_list'],
                              exp_gmix,
                              dev_gmix,
                              lm_max_try=self.obj_ntry)
        res=gm.get_result()
        self.data['cmodel_flags'][index] = res['flags']
        self.data['cmodel_iter'][index] = res['numiter']
        self.data['cmodel_tries'][index] = res['ntry']

        if res['flags']==0:
            f=res['fracdev']
            ferr=res['fracdev_err']
            self.data['frac_dev'][index] = f
            self.data['frac_dev_err'][index] = ferr
            flux=(1.-f)*self.data['exp_flux'][index] \
                    + f*self.data['dev_flux'][index]
            flux_err2=(1.-f)**2*self.data['exp_flux_err'][index]**2 \
                         + f**2*self.data['dev_flux_err'][index]**2
            flux_err=sqrt(flux_err2)
            self.data['cmodel_flux'][index] = flux
            self.data['cmodel_flux_err'][index] = flux_err

            if self.debug:
                fmt='\t\t%s: %g +/- %g'
                print >>stderr,fmt % ('frac_dev',f,ferr)
                print >>stderr,fmt % ('cmodel_flux',flux,flux_err)
                    
    def _fit_psf_flux(self, index, sdata):
        if self.debug:
            print >>stderr,'\tfitting psf flux'
        gm=GMixFitMultiPSFFlux(sdata['imlist'],
                               sdata['wtlist'],
                               sdata['jacob_list'],
                               sdata['psf_gmix_list'],
                               lm_max_try=self.obj_ntry)
        res=gm.get_result()
        self.data['psf_flags'][index] = res['flags']
        self.data['psf_iter'][index] = res['numiter']
        self.data['psf_tries'][index] = res['ntry']

        if res['flags']==0:
            self.data['psf_pars'][index,:]=res['pars']
            self.data['psf_pars_cov'][index,:,:] = res['pcov']

            flux=res['pars'][2]
            flux_err=sqrt(res['pcov'][2,2])
            self.data['psf_flux'][index] = flux
            self.data['psf_flux_err'][index] = flux_err

            n=get_model_names('psf')
            for sn in _stat_names:
                self.data[n[sn]][index] = res[sn]

            if self.debug:
                fmt='\t\t%s: %g +/- %g'
                print >>stderr,fmt % ('psf_flux',flux,flux_err)

    def _fit_match(self, index, sdata):
        if self.debug:
            print >>stderr,'\tfitting matched flux'
        niter=0
        ntry=0
        flux=DEFVAL
        flux_err=PDEFVAL
        if self.det_cat is None:
            # this is the detection band, just copy some data
            flags,pars,pcov,niter0,ntry0,mod=\
                    self._get_best_simple_pars(self.data,index)
            if flags==0:
                niter=niter0
                ntry=ntry0
                flux=pars[5]
                flux_err=sqrt(pcov[5,5])
        else:
            flags,pars0,pcov0,niter0,ntry0,mod=\
                    self._get_best_simple_pars(self.det_cat,index)
            # if flags != 0 it is because we could not find a good fit of any
            # model
            if flags==0:
                match_gmix = gmix_image.GMix(pars0, type=mod)

                gm=GMixFitMultiMatch(sdata['imlist'],
                                     sdata['wtlist'],
                                     sdata['jacob_list'],
                                     sdata['psf_gmix_list'],
                                     match_gmix,
                                     lm_max_try=self.obj_ntry)
                res=gm.get_result()
                flags=res['flags']
                if flags==0:
                    flux=res['F']
                    flux_err=res['Ferr']
                    niter=res['numiter']
                    ntry=res['ntry']

        self.data['match_flags'][index] = flags
        self.data['match_model'][index] = mod
        self.data['match_iter'][index] = niter
        self.data['match_tries'][index] = ntry
        self.data['match_flux'][index] = flux
        self.data['match_flux_err'][index] = flux_err
        if self.debug:
            fmt='\t\t%s[%s]: %g +/- %g'
            print >>stderr,fmt % ('match_flux',mod,flux,flux_err)


    def _get_best_simple_pars(self, data, index):
        expflags=data['exp_flags'][index]
        devflags=data['dev_flags'][index]

        flags=0
        if expflags==0 and devflags==0:
            if (data['exp_loglike'][index] 
                    > data['dev_loglike'][index]):
                mod='exp'
            else:
                mod='dev'
        elif expflags==0:
            mod='exp'
        elif devflags==0:
            mod='dev'
        else:
            flags |= (EXP_FIT_FAILURE+DEV_FIT_FAILURE)
            return flags,DEFVAL,PDEFVAL,0,0,'nil'

        pn='%s_pars' % mod
        pcn='%s_pars_cov' % mod
        itn='%s_iter' % mod
        tn='%s_tries' % mod

        pars=data[pn][index]
        pcov=data[pcn][index]
        niter=data[itn][index]
        ntry=data[tn][index]

        return flags,pars,pcov,niter,ntry,mod

    def _get_imlists_outlier_reject(self, index):
        """
        Get the image lists.

        Remove 10-sigma outliers from the regions with weight
        by setting their weight to zero
        """
        mosaic0=self.meds.get_mosaic(index,type='image')
        wt_mosaic0=self.meds.get_cweight_mosaic(index)

        # cut out the coadd
        box_size  = mosaic0.shape[1]
        mosaic    = mosaic0[box_size:, :].copy()
        wt_mosaic = wt_mosaic0[box_size:, :].copy()

        # do outlier rejection on the pixels with weight
        wtmax=wt_mosaic.max()
        keep_logic = ( wt_mosaic > 0.2*wtmax )
        wkeep=numpy.where(keep_logic)

        if wkeep[0].size > 0:
            mos_keep=mosaic[wkeep]
            crap,sig=sigma_clip(mos_keep.ravel())
            med=numpy.median(mos_keep)

            # note repeating the weight cut
            wout=numpy.where(  keep_logic
                             & (numpy.abs(mosaic-med) > self.pix_nsig*sig) )
            if wout[0].size > 0:
                print >>stderr,'\tfound %d %d-sigma outliers' % (wout[0].size,self.pix_nsig)
                wt_mosaic[wout] = 0.0

        imlist=meds.split_mosaic(mosaic)
        wtlist=meds.split_mosaic(wt_mosaic)

        import images
        images.view_mosaic(imlist)
        stop
        return imlist, wtlist


    def _get_jacobian_list(self, index):
        """
        Get a list of the jocobians for this object
        skipping the coadd
        """
        jacob_list=self.meds.get_jacobian_list(index)
        jacob_list=jacob_list[1:]
        return jacob_list
    
    '''
    def _get_cenlist(self, index):
        """
        Get the median of the cutout row,col centers,
        skipping the coadd
        """
        ncut=self.meds['ncutout'][index]
        cenlist=[]
        # start at 1 to skip coadd
        for icut in xrange(1,ncut):

            row0 = self.meds['cutout_row'][index,icut]
            col0 = self.meds['cutout_col'][index,icut]
            cen0 = [row0, col0]
            cenlist.append(cen0)

        return cenlist
    '''

    def _get_psfex_reclist(self, index):
        """
        Generate psfex reconstructions for the SE images
        associated with the cutouts, skipping the coadd

        add a little noise for the fitter
        """
        ncut=self.meds['ncutout'][index]
        imlist=[]
        ivarlist=[]
        cenlist=[]
        siglist=[]
        flist=[]
        for icut in xrange(1,ncut):
            file_id=self.meds['file_id'][index,icut]
            pex=self.psfex_list[file_id]
            fname=pex['filename']

            row=self.meds['orig_row'][index,icut]
            col=self.meds['orig_col'][index,icut]

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



    def _load_all_psfex_objects(self):
        """
        Load psfex objects for each of the SE images
        include the coadd so we get  the index right
        """
        psfex_list=[]
        info=self.meds.get_image_info()
        nimage=info.size

        for i in xrange(nimage):
            impath=info['image_path'][i].strip()
            psfpath=impath.replace('.fits.fz','_psfcat.psf')
            pex=psfex.PSFEx(psfpath)
            psfex_list.append(pex)

        self.psfex_list=psfex_list


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
            end=self.meds.size-1
        else:
            start=self.obj_range[0]
            end=self.obj_range[1]

        self.index_list = numpy.arange(start,end+1)

    def _set_det_cat(self, det_cat):
        if det_cat is not None:
            if det_cat.size != self.meds.size:
                mess=("det_cat should be collated and match full "
                      "coadd size (%d) got %d")
                mess=mess % (self.meds.size,det_cat.size)
                raise ValueError(mess)
        self.det_cat=det_cat

    def _make_struct(self):
        nobj=self.meds.size

        dt=[('id','i4'),
            ('flags','i4'),
            ('nimage','i4'),
            ('time','f8')]

        simple_npars=6
        simple_models=['exp','dev']
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

            data[n['s2n_w']] = DEFVAL
            data[n['loglike']] = BIG_DEFVAL
            data[n['chi2per']] = PDEFVAL
            data[n['aic']] = BIG_PDEFVAL
            data[n['bic']] = BIG_PDEFVAL
        
        self.data=data


    def _get_imlist(self, index, type='image'):
        """
        get the image list, skipping the coadd
        """
        imlist=self.meds.get_cutout_list(index,type=type)
        imlist=imlist[1:]
        return imlist

    def _get_wtlist(self, index):
        """
        get the weight list.

        If using the seg map, mark pixels outside the coadd object region as
        zero weight
        """
        wtlist=self.meds.get_cweight_cutout_list(index)
        wtlist=wtlist[1:]
        return wtlist



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
           'flux',
           'flux_err',
           'g',
           'g_cov',
           'iter',
           'tries']
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


