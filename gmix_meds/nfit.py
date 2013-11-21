import os
from sys import stderr,stdout
import numpy
import meds
import psfex
import ngmix

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

PSF1_NOT_KEPT=2**7

NO_ATTEMPT=2**30

PSF_S2N=1.e6
PSF_OFFSET_MAX=0.25
EM_MAX_TRY=3

SIMPLE_MODELS_DEFAULT = ['exp','dev']


class MedsFit(object):
    def __init__(self, meds_files, **keys):
        """
        Model fitting

        parameters
        ----------
        meds_file:
            string of meds path, or list thereof for different bands
        fit_types: list of strings
            'psf','simple'
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
        checkpoint_data: array of fields, optional
            The data representing a previous checkpoint
        """

        self.conf={}
        self.conf.update(keys)

        self.nwalkers=keys.get('nwalkers',20)
        self.burnin=keys.get('burnin',400)
        self.nstep=keys.get('nstep',200)
        self.do_pqr=keys.get("do_pqr",False)
        self.do_lensfit=keys.get("do_lensfit",False)
        self.mca_a=keys.get('mca_a',2.0)
        self.g_prior=keys.get('g_prior',None)


        self.meds_files=_get_as_list(meds_files)
        self.checkpoint = self.conf.get('checkpoint',172800)
        self.checkpoint_file = self.conf.get('checkpoint_file',None)
        self._checkpoint_data=keys.get('checkpoint_data',None)

        self.nband=len(meds_files)
        self.iband = range(self.nband)

        self._load_meds_files()
        self.mb_psfex_list = self._get_all_psfex_objects()

        self.obj_range=keys.get('obj_range',None)
        self._set_index_list()

        # in arcsec (or units of jacobian)
        self.use_cenprior=keys.get("use_cenprior",True)
        self.cen_width=keys.get('cen_width',1.0)

        self.psf_model=keys.get('psf_model','em2')
        self.psf_offset_max=keys.get('psf_offset_max',PSF_OFFSET_MAX)
        self.psf_ngauss=get_psf_ngauss(self.psf_model)

        self.debug=keys.get('debug',0)

        self.psf_ntry=keys.get('psf_ntry', EM_MAX_TRY)

        self.region=keys.get('region','seg_and_sky')
        self.max_box_size=keys.get('max_box_size',2048)

        self.simple_models=keys.get('simple_models',SIMPLE_MODELS_DEFAULT )

        self.reject_outliers=keys.get('reject_outliers',False) # from cutouts

        self.make_plots=keys.get('make_plots',False)
        self.prompt=keys.get('prompt',True)


        if self._checkpoint_data is not None:
            self.data=self._checkpoint_data
        else:
            self._make_struct()

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

    def _get_all_psfex_objects(self):
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

            psfex_list = self._get_psfex_objects(meds)
            mb_psfex_list.append( psfex_list )

        return mb_psfex_list

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
            end=self.nobj-1
        else:
            start=self.obj_range[0]
            end=self.obj_range[1]

        self.index_list = numpy.arange(start,end+1)


_stat_names=['s2n_w',
             'lnprob',
             'chi2per',
             'dof',
             'aic',
             'bic']


_psf_ngauss_map={'em1':1, 'em2':2, 'em3':3}
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


