import numpy
from numpy.random import randn
import fitsio
import meds
import gmix_image
from gmix_image.gmix_fit import GMixFitPSFJacob
import psfex

NO_SE_CUTOUTS=2**0
PSF_FIT_FAILURE=2**1
PSF_S2N=1.e6

class MedsFit(object):
    def __init__(self, meds_file,
                 psf_ngauss=2,
                 use_seg=True,
                 det_cat=None):
        self.meds_file=meds_file
        self.meds=meds.MEDS(meds_file)
        self.nobj=self.meds.size

        self.psf_ngauss=psf_ngauss
        self.use_seg=use_seg
        self.det_cat=det_cat

        self._load_all_psfex_objects()

    def fit_objs(self, start, end):
        """
        Fit objects in the indicated range
        """
        for index in xrange(start,end+1):
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

        imlist=self._get_imlist(index)
        wtlist=self._get_wtlist(index)
        jacob_list=self._get_jacobian_list(index)
        cen0=self._get_cen0(index)

        psf_gmix_list=self._get_psf_gmix_list(index,jacob_list)
        if psf_gmix_list is None:
            self.data['flags'][index] |= PSF_FIT_FAILURE
            return

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

        If using the seg map, mark pixels outside the
        object as zero weight
        """
        wtlist=self._get_imlist(index, type='weight')

        if self.use_seg:
            seglist=self._get_imlist(index,type='seg')
            ii = index+1
            for wt,seg in zip(wtlist,seglist):
                w=numpy.where(seglist != ii)
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
        row0 = numpy.median(self.meds['cutout_row'][index,1:])
        col0 = numpy.median(self.meds['cutout_col'][index,1:])

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

    def _get_psf_gmix_list(self,index,jacob_list):
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

    def _make_struct(self):
        nobj=self.meds.size

        dt=[('id','i4'), ('flags','i4')]

        data=numpy.zeros(nobj, dtype=dt)
        
        self.data=data

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
