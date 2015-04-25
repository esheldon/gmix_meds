from __future__ import print_function

import numpy
from numpy import where, zeros
import fitsio
import meds
import ngmix
from ngmix import Observation, ObsList, MultiBandObsList, Jacobian
from . import files
from .nfit import MedsFit

class MissingFit(Exception):
    """
    EM algorithm hit max iter
    """
    def __init__(self, value):
         self.value = value
    def __str__(self):
        return repr(self.value)


class RoundModelBurner(dict):
    """
    Run through a collated file and comput the round model as well
    as s2n_r, T_s2n_r
    """
    def __init__(self, config_file, collated_file):
        conf=files.read_yaml(config_file)
        self.collated_file=collated_file
        self.update(conf)

        self.load()
        self.set_rev()

    def go(self):
        """
        loop over objects, load data, create the round
        info, copy to the output
        """
        nobj = self.model_fits.size

        for i in xrange(nobj):
            print("%d:%d" % (i+1,nobj))

            self.process_obj(i)

    def process_obj(self, index):

        try:
            mbo=self.get_mb_obs_list(index)
        except MissingFit as e:
            print("   skipping object: '%s'" % str(e))


    def get_mb_obs_list(self, index):
        """
        raise MissingFit if any bands missing or no cutouts
        """
        mbo=MultiBandObsList()

        rev=self.rev_number

        ncut=self.h_number[index]
        print("    number:",index+1,"total cutouts:",ncut)

        i=index
        if rev[i] != rev[i+1]:
            w=rev[ rev[i]:rev[i+1] ]

            ed=self.epoch_data[w]

            for band in xrange(self.nband):
                obslist=self.get_band_obslist(index, band, ed)

                mbo.append( obslist )
        else:
            raise MissingFit("band missing")

        return mbo

    def get_band_obslist(self, index, band, epoch_data):
        w,=where(  (epoch_data['cutout_index'] > 0)
                 & (epoch_data['psf_fit_flags']==0)
                 & (epoch_data['band_num']==band) )

        print("        found %d good SE cutouts for band %d" % (w.size, band))
        if w.size == 0:
            # we don't process objects without one of the bands
            raise MissingFit("SE band missing")

        obslist = ObsList()

        for ib in xrange(w.size):
            ii = w[ib]
            ed = epoch_data[ii]

            obs = self.get_obs(index, band, ed)

            obslist.append(obs)

        return obslist

    def get_obs(self, index, band, epoch_data):
        """
        We load the psf gaussian mixture into a fake
        psf observation

        get the observation.  The image is not used,
        so is faked

        epoch_data is a single struct
        """

        cutid = epoch_data['cutout_index']

        m=self.meds_list[band]
        wt = m.get_cutout(index, cutid, type='weight')

        jdict = m.get_jacobian(index, cutid)
        jacob=Jacobian(jdict['row0'],
                       jdict['col0'],
                       jdict['dudrow'],
                       jdict['dudcol'],
                       jdict['dvdrow'],
                       jdict['dvdcol'])

        imfake = wt*0

        psf_pars = epoch_data['psf_fit_pars']
        psf_gmix = ngmix.GMix(pars=psf_pars)

        psf_im_fake=zeros( (1,1) )
        psf_obs = Observation(psf_im_fake, gmix=psf_gmix)

        obs = Observation(imfake, weight=wt, jacobian=jacob, psf=psf_obs)
        return obs


    def load(self):
        print("loading data from:",self.collated_file)
        with fitsio.FITS(self.collated_file) as fits:
            print("    loading model fits")
            self.model_fits=fits['model_fits'].read()
            print("    loading epoch data")
            self.epoch_data=fits['epoch_data'].read()
            print("    loading meta data")
            self.meta=fits['meta_data'].read()

        meds_fnames=self.meta['meds_file']
        self.nband=len(meds_fnames)

        self.meds_list=[]
        for fname in meds_fnames:
            print("loading MEDS file:",fname)
            m=meds.MEDS(fname.strip())
            self.meds_list.append( m )

    def set_rev(self):
        from esutil.stat import histogram

        print("histogramming epoch 'number'")

        m=self.meds_list[0]
        number_max=m['number'].max()

        h_number,rev = histogram(self.epoch_data['number'],
                                 min=1,
                                 max=number_max,
                                 rev=True)

        self.h_number=h_number
        self.rev_number=rev
