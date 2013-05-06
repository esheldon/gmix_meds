from sys import stderr
import numpy
from numpy import sqrt
from numpy.random import randn
import fitsio
import meds
import gmix_image
from gmix_image.gmix_fit import GMixFitPSFJacob,\
        GMixFitMultiSimple,GMixFitMultiCModel, \
        GMixFitMultiPSFFlux
import psfex

DEFVAL=-9999
PDEFVAL=9999

NO_SE_CUTOUTS=2**0
PSF_FIT_FAILURE=2**1
EXP_FIT_FAILURE=2**2
DEV_FIT_FAILURE=2**3

PSF_S2N=1.e6

class MedsFit(object):
    def __init__(self, meds_file,
                 obj_range=None,
                 psf_ngauss=2,
                 use_seg=True,
                 det_cat=None,
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
        psf_ngauss: optional,int
            Number of gaussians to fit to PSF.
        use_seg: bool,optional
            If True, limit pixels to the seg map.  We just ask which pixels
            have same seg id as the center one.  This is far from optimal!
            Somehow need overall seg map in ra,dec space, maybe interpolate
            from coadd
        det_cat: optional
            Catalog to use as "detection" catalog; an overall flux will be fit
            with best simple model fit from this.

        TODO:
            psfflux
            cmodel
            fluxonly
        """

        self.meds_file=meds_file
        self.meds=meds.MEDS(meds_file)
        self.nobj=self.meds.size

        self.obj_range=obj_range
        self.psf_ngauss=psf_ngauss
        self.use_seg=use_seg
        self.det_cat=det_cat
        self.debug=debug

        self.simple_models=['exp','dev']

        self._make_struct()
        self._load_all_psfex_objects()

    def get_data(self):
        idlist=self.get_index_list()
        return self.data[idlist]

    def get_index_list(self):
        """
        Return a list of indices to be processed
        """
        if self.obj_range is None:
            start=0
            end=self.meds.size-1
        else:
            start=self.obj_range[0]
            end=self.obj_range[1]

        return numpy.arange(start,end+1)


    def do_fits(self):
        """
        Fit objects in the indicated range
        """
        idlist=self.get_index_list()
        last=idlist[-1]
        for index in idlist:
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

        cen0=self._get_cen0(index)
        imlist=self._get_imlist(index)
        wtlist=self._get_wtlist(index,cen0)
        jacob_list=self._get_jacobian_list(index)

        if self.debug: print >>stderr,'\tfitting psfs'

        psf_gmix_list=self._fit_psfs(index,jacob_list)
        if psf_gmix_list is None:
            self.data['flags'][index] |= PSF_FIT_FAILURE
            return

        sdata={'imlist':imlist,'wtlist':wtlist,
               'jacob_list':jacob_list,'cen0':cen0,
               'psf_gmix_list':psf_gmix_list}

        self._fit_simple_models(index, sdata)
        self._fit_cmodel(index, sdata)
        self._fit_psf_flux(index, sdata)

        if self.debug >= 3:
            self._debug_image(sdata['imlist'][0],sdata['wtlist'][-1])

    def _fit_psfs(self,index,jacob_list):
        """
        Generate psfex images for all SE images and fit
        them to gaussian mixture models
        """
        imlist,ivarlist,cenlist=self._get_psfex_reclist(index)
        gmix_list=[]

        for i in xrange(len(imlist)):
            im=imlist[i]
            ivar=ivarlist[i]
            jacob=jacob_list[i]
            cen0=cenlist[i]

            gm_psf=GMixFitPSFJacob(im,
                                   ivar,
                                   jacob,
                                   cen0,
                                   self.psf_ngauss)
            res=gm_psf.get_result()
            if res['flags'] != 0:
                print 'error: psf fitting failed, '
                print res
                return None

            gmix_psf=gm_psf.get_gmix()
            gmix_list.append( gmix_psf )

        return gmix_list

    def _fit_simple_models(self, index, sdata):
        """
        Fit all the simple models
        """
        if self.debug:
            print >>stderr,'\tfitting simple models'

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

        if rfc_res['flags']==0:
            self.data[n['rfc_pars']][index,:] = rfc_res['pars']
            self.data[n['rfc_pars_cov']][index,:] = rfc_res['pcov']

        self.data[n['flags']][index] = res['flags']

        if res['flags'] == 0:
            self.data[n['pars']][index,:] = res['pars']
            self.data[n['iter']][index] = res['numiter']
            self.data[n['pars_cov']][index,:,:] = res['pcov']

            flux=res['pars'][5]
            flux_err=sqrt(res['pcov'][5,5])
            self.data[n['flux']][index] = flux
            self.data[n['flux_err']][index] = flux_err

            self.data[n['g']][index,:] = res['pars'][2:2+2]
            self.data[n['g_cov']][index,:,:] = res['pcov'][2:2+2,2:2+2]
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
                              sdata['cen0'],
                              model)
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
                              sdata['cen0'],
                              exp_gmix,
                              dev_gmix)
        res=gm.get_result()
        self.data['cmodel_flags'][index] = res['flags']
        self.data['cmodel_iter'][index] = res['numiter']

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
                               sdata['cen0'])
        res=gm.get_result()
        self.data['psf_flags'][index] = res['flags']
        self.data['psf_iter'][index] = res['numiter']

        if res['flags']==0:
            self.data['psf_pars'][index,:]=res['pars']
            self.data['psf_pars_cov'][index,:,:] = res['pcov']

            flux=res['pars'][2]
            flux_err=sqrt(res['pcov'][2,2])
            self.data['psf_flux'][index] = flux
            self.data['psf_flux_err'][index] = flux_err
            if self.debug:
                fmt='\t\t%s: %g +/- %g'
                print >>stderr,fmt % ('psf_flux',flux,flux_err)

    def _get_imlist(self, index, type='image'):
        """
        get the image list, skipping the coadd
        """
        imlist=self.meds.get_cutout_list(index,type=type)
        imlist=imlist[1:]
        return imlist

    def _get_wtlist(self, index, cen0):
        """
        get the weight list.

        If using the seg map, mark pixels outside the
        object as zero weight

        have kludge checking seg val against the
        center
        """
        wtlist=self._get_imlist(index, type='weight')

        if self.use_seg:
            seglist=self._get_imlist(index,type='seg')
            for wt,seg in zip(wtlist,seglist):
                sval=seg[cen0[0], cen0[1]]
                w=numpy.where(seg != sval)
                if w[0].size > 0:
                    wt[w] = 0.0

        return wtlist

    def _get_jacobian_list(self, index):
        """
        Get a list of the jocobians for this object
        skipping the coadd
        """
        jacob_list=self.meds.get_jacobian_list(index)
        jacob_list=jacob_list[1:]
        return jacob_list
    
    def _get_cen0(self, index):
        """
        Get the median of the cutout row,col centers,
        skipping the coadd
        """
        ncut=self.meds['ncutout'][index]
        row0 = numpy.median(self.meds['cutout_row'][index,1:ncut])
        col0 = numpy.median(self.meds['cutout_col'][index,1:ncut])

        return [row0,col0]

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
        for icut in xrange(1,ncut):
            file_id=self.meds['file_id'][index,icut]
            pex=self.psfex_list[file_id]

            row=self.meds['orig_row'][index,icut]
            col=self.meds['orig_col'][index,icut]

            im0=pex.get_rec(row,col)
            cen0=pex.get_center(row,col)

            im,skysig=add_noise_matched(im0, PSF_S2N)
            ivar=1./skysig**2

            imlist.append( im )
            ivarlist.append(ivar)
            cenlist.append(cen0)

        return imlist, ivarlist, cenlist



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
            nm='-rfc %s' % n['flux']
            flux=rfc_res['pars'][1]
            flux_err=sqrt(rfc_res['pcov'][1,1])
            print >>stderr,fmt % (nm,flux,flux_err)

            nm='-rfc T'
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

    def _make_struct(self):
        nobj=self.meds.size

        dt=[('id','i4'),
            ('flags','i4')]

        simple_npars=6
        simple_models=['exp','dev']
        for model in simple_models:
            n=get_model_names(model)

            dt+=[(n['rfc_flags'],'i4'),
                 (n['rfc_iter'],'i4'),
                 (n['rfc_pars'],'f8',2),
                 (n['rfc_pars_cov'],'f8',(2,2)),
                 (n['flags'],'i4'),
                 (n['iter'],'i4'),
                 (n['pars'],'f8',simple_npars),
                 (n['pars_cov'],'f8',(simple_npars,simple_npars)),
                 (n['flux'],'f8'),
                 (n['flux_err'],'f8'),
                 (n['g'],'f8',2),
                 (n['g_cov'],'f8',(2,2)),
                
                 (n['s2n_w'],'f8'),
                 (n['loglike'],'f8'),
                 (n['chi2per'],'f8'),
                 (n['dof'],'i4'),
                 (n['fit_prob'],'f8'),
                 (n['aic'],'f8'),
                 (n['bic'],'f8'),
                ]

        dt += [('cmodel_flags','i4'),
               ('cmodel_iter','i4'),
               ('cmodel_flux','f8'),
               ('cmodel_flux_err','f8'),
               ('frac_dev','f8'),
               ('frac_dev_err','f8'),
               ('psf_flags','i4'),
               ('psf_iter','i4'),
               ('psf_pars','f8',3),
               ('psf_pars_cov','f8',(3,3)),
               ('psf_flux','f8'),
               ('psf_flux_err','f8')]


        data=numpy.zeros(nobj, dtype=dt)
        data['id'] = 1+numpy.arange(nobj)
        data['frac_dev'] = DEFVAL
        data['frac_dev_err'] = PDEFVAL
        data['cmodel_flux'] = DEFVAL
        data['cmodel_flux_err'] = PDEFVAL

        data['psf_pars'] = DEFVAL
        data['psf_pars_cov'] = PDEFVAL
        data['psf_flux'] = DEFVAL
        data['psf_flux_err'] = PDEFVAL

        for model in simple_models:
            pars_name='%s_pars' % model
            cov_name='%s_cov' % model

            n=get_model_names(model)
            data[n['rfc_pars']] = DEFVAL
            data[n['rfc_pars_cov']] = PDEFVAL
            data[n['pars']] = DEFVAL
            data[n['pars_cov']] = PDEFVAL
            data[n['flux']] = DEFVAL
            data[n['flux_err']] = PDEFVAL
            data[n['g']] = DEFVAL
            data[n['g_cov']] = PDEFVAL

            data[n['s2n_w']] = DEFVAL
            data[n['loglike']] = -9.999e9
            data[n['chi2per']] = PDEFVAL
            data[n['aic']] = DEFVAL
            data[n['bic']] = DEFVAL
        
        self.data=data

def get_model_names(model):
    names=['rfc_flags',
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
           's2n_w',
           'loglike',
           'chi2per',
           'dof',
           'fit_prob',
           'aic',
           'bic']

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
