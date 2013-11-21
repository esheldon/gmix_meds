import os
from sys import stderr,stdout
import time
import numpy
import meds
import psfex
import ngmix
from ngmix import srandu

from ngmix import GMixMaxIterEM

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

        self.nband=len(self.meds_files)
        self.iband = range(self.nband)

        self._load_meds_files()
        self.psfex_lol = self._get_psfex_lol()

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
        self.psf_maxiter=keys.get('psf_maxiter', EM_MAX_ITER)
        self.psf_tol=keys.get('psf_tol', PSF_TOL)

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
        print >>stderr,"time per:",tm/num


    def fit_obj(self, dindex):
        """
        Process the indicated object through the requested fits
        """

        t0=time.time()

        # for checkpointing
        self.data['processed'][dindex]=1

        mindex = self.index_list[dindex]

        self.data['id'][dindex] = self.meds_list[0]['number'][mindex]

        self.data['flags'][dindex] = self._obj_check(mindex)
        if self.data['flags'][dindex] != 0:
            return 0

        # lists of lists
        im_lol,wt_lol,self.coadd_lol = self._get_imlol_wtlol(dindex,mindex)
        jacob_lol=self._get_jacobian_lol(mindex)

        print >>stderr,im_lol[0][0].shape
    
        keep_lol,psf_gmix_lol,flags=self._fit_psfs(mindex,jacob_lol)
        if any(flags):
            self.data['flags'][dindex] = PSF_FIT_FAILURE 
            return
        stop

        im_lol, wt_lol, jacob_lol, len_list = \
            self._extract_sub_lists(keep_lol,im_lol,wt_lol,jacob_lol)
        

        self.data['nimage_use'][dindex, :] = len_list

        sdata={'keep_lol':keep_lol,
               'im_lol':im_lol,
               'wt_lol':wt_lol,
               'jacob_lol':jacob_lol,
               'psf_gmix_lol':psf_gmix_lol}

        self._do_all_fits(dindex, sdata)

        self.data['time'][dindex] = time.time()-t0

    def _obj_check(self, mindex):
        for band in self.iband:
            meds=self.meds_list[band]
            flags=self._obj_check_one(meds, mindex)
            if flags != 0:
                break
        return flags

    def _obj_check_one(self, meds, mindex):
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

        for band in self.iband:
            meds=self.meds_list[band]

            # inherited functions
            imlist,coadd=self._get_imlist(meds,mindex)
            wtlist=self._get_wtlist(meds,mindex)

            self.data['nimage_tot'][dindex,band] = len(imlist)

            im_lol.append(imlist)
            wt_lol.append(wtlist)
            coadd_lol.append(coadd)

        
        return im_lol,wt_lol, coadd_lol

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


    def _get_jacobian_lol(self, mindex):
        """
        Get a list of the jocobians for this object
        skipping the coadd
        """

        mb_jacob_list=[]
        for band in self.iband:
            meds=self.meds_list[band]

            jacob_list = self._get_jacobian_list(meds,mindex)
            mb_jacob_list.append(jacob_list)

        return mb_jacob_list

    def _get_jacobian_list(self, meds, mindex):
        """
        Get a list of the jocobians for this object
        skipping the coadd
        """
        jlist0=meds.get_jacobian_list(mindex)

        jlist=[]
        for jdict in jlist0:
            #print jdict
            j=ngmix.Jacobian(jdict['row0'],
                             jdict['col0'],
                             jdict['dudrow'],
                             jdict['dudcol'],
                             jdict['dvdrow'],
                             jdict['dvdcol'])
            jlist.append(j)

        jlist=jlist[1:]
        return jlist


    def _fit_psfs(self, mindex, jacob_lol):
        """
        fit psfs for all bands
        """
        keep_lol=[]
        gmix_lol=[]
        flags_list=[]

        for band in self.iband:
            meds=self.meds_list[band]
            jacob_list=jacob_lol[band]
            psfex_list=self.psfex_lol[band]

            keep_list, gmix_list, flags = self._fit_psfs_oneband(meds,mindex,jacob_list,psfex_list)

            keep_lol.append( keep_list )
            gmix_lol.append( gmix_list )

            # only care about flags if we have no psfs left
            if len(keep_list) == 0:
                flags_list.append( flags )
            else:
                flags_list.append( 0 )

       
        return keep_lol, gmix_lol, flags


    def _fit_psfs_oneband(self,meds,mindex,jacob_list,psfex_list):
        """
        Generate psfex images for all SE images and fit
        them to gaussian mixture models
        """
        ptuple = self._get_psfex_reclist(meds, psfex_list, mindex)
        imlist,cenlist,siglist,flist,cenpix=ptuple

        keep_list=[]
        gmix_list=[]

        flags=0

        for i in xrange(len(imlist)):
            im=imlist[i]
            jacob0=jacob_list[i]
            sigma=siglist[i]

            cen0=cenlist[i]
            # the dimensions of the psfs are different, need
            # new center
            jacob=jacob0.copy()
            jacob._data['row0'] = cen0[0]
            jacob._data['col0'] = cen0[1]

            try:
                fitter=self._do_fit_psf(im,jacob,sigma)

                gmix_psf=fitter.get_gmix()
                if self._should_keep_psf(gmix_psf):
                    gmix_list.append( gmix_psf )
                    keep_list.append(i)
                else:
                    flags |= PSF_LARGE_OFFSETS 

            except GMixMaxIterEM:
                print >>stderr,'psf fail',flist[i]


        return keep_list, gmix_list, flags


    def _get_psfex_reclist(self, meds, psfex_list, mindex):
        """
        Generate psfex reconstructions for the SE images
        associated with the cutouts, skipping the coadd

        add a little noise for the fitter
        """
        ncut=meds['ncutout'][mindex]
        imlist=[]
        cenlist=[]
        siglist=[]
        flist=[]
        for icut in xrange(1,ncut):
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

        return imlist, cenlist, siglist, flist, [row,col]

    def _should_keep_psf(self, gm):
        keep=True
        if self.psf_ngauss == 2:
            offset_arcsec = calc_offset_arcsec(gm)
            if offset_arcsec > self.psf_offset_max:
                keep=False

        return keep

    def _do_fit_psf(self, im, jacob, sigma_guess):
        """
        Fit a single psf
        """
        s2=sigma_guess**2
        im_with_sky, sky = ngmix.em.prep_image(im)

        fitter=ngmix.em.GMixEM(im_with_sky, jacobian=jacob)

        for i in xrange(self.psf_ntry):

            s2guess=s2*jacob._data['det'][0]

            gm_guess=self._get_em_guess(s2guess)
            print 'guess:'
            print gm_guess
            try:
                fitter.go(gm_guess, sky,
                          maxiter=self.psf_maxiter,
                          tol=self.psf_tol)
                break
            except GMixMaxIterEM:
                res=fitter.get_result()
                print 'last fit:'
                print fitter.get_gmix()
                print 'try:',i+1,'fdiff:',res['fdiff'],'numiter:',res['numiter']
                if i == (self.psf_ntry-1):
                    raise

        return fitter

    def _get_em_guess(self, sigma2):
        """
        Guess for the EM algorithm
        """
        if self.psf_ngauss==1:
            pars=numpy.array( [1.0, 0.0, 0.0, 
                               0.5*sigma2*(1.0 + 0.1*srandu()),
                               0.0,
                               0.5*sigma2*(1.0 + 0.1*srandu())] )
        else:
            em2_fguess=numpy.array([0.5793612389470884,1.621860687127999])
            em2_pguess=numpy.array([0.596510042804182,0.4034898268889178])

            pars=numpy.array( [em2_pguess[0],
                               0.1*srandu(),
                               0.1*srandu(),
                               0.5*em2_fguess[0]*sigma2*(1.0 + 0.1*srandu()),
                               0.0,
                               0.5*em2_fguess[0]*sigma2*(1.0 + 0.1*srandu()),

                               em2_pguess[1],
                               0.1*srandu(),
                               0.1*srandu(),
                               0.5*em2_fguess[1]*sigma2*(1.0 + 0.1*srandu()),
                               0.0,
                               0.5*em2_fguess[1]*sigma2*(1.0 + 0.1*srandu())] )

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

    def _get_psfex_lol(self):
        """
        Load psfex objects for each of the SE images
        include the coadd so we get  the index right
        """
        print 'loading psfex'
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
            end=self.nobj-1
        else:
            start=self.obj_range[0]
            end=self.obj_range[1]

        self.index_list = numpy.arange(start,end+1)

    def _make_struct(self):
        nband=self.nband
        bshape=(nband,)

        dt=[('id','i4'),
            ('processed','i1'),
            ('flags','i4'),
            ('nimage_tot','i4',bshape),
            ('nimage_use','i4',bshape),
            ('time','f8')]

        
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
                 (n['iter'],'i4'),
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
                ]
            if self.do_lensfit:
                dt += [(n['g_sens'], 'f8', 2)]
            if self.do_pqr:
                dt += [(n['P'], 'f8'),
                       (n['Q'], 'f8', 2),
                       (n['R'], 'f8', (2,2))]


        # the psf fits are done for each band separately
        n=get_model_names('psf')
        dt += [(n['flags'],   'i4',bshape),
               (n['flux'],    'f8',bshape),
               (n['flux_err'],'f8',bshape)]

        num=self.index_list.size
        data=numpy.zeros(num, dtype=dt)

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

            if self.do_lensfit:
                data[n['g_sens']] = DEFVAL
            if self.do_pqr:
                data[n['P']] = DEFVAL
                data[n['Q']] = DEFVAL
                data[n['R']] = DEFVAL

        data['psf_flags'] = NO_ATTEMPT
        data['psf_flux'] = DEFVAL
        data['psf_flux_err'] = PDEFVAL
     
        self.data=data

_stat_names=['s2n_w',
             'lnprob',
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

def calc_offset_arcsec(gm):
    data=gm.get_data()

    offset=sqrt( (data['row'][0]-data['row'][1])**2 + 
                 (data['col'][0]-data['col'][1])**2 )
    offset_arcsec=offset*PIXEL_SCALE
    return offset_arcsec


