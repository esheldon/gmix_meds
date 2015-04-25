from __future__ import print_function

import numpy
from numpy import where, zeros, sqrt, median
from numpy.linalg import LinAlgError
import fitsio
import meds
import ngmix
from ngmix import Observation, ObsList, MultiBandObsList, Jacobian
from ngmix import GMixRangeError
from . import files
from .util import Namer
from .nfit import MedsFit, NO_ATTEMPT, DEFVAL

MISSING_FIT=2**9
BAD_MODEL=2**10
TS2N_FAIL=2**11

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

        self['use_logpars'] = self.get('use_logpars',False)

        self.load()
        self.set_rev()
        self.set_struct()
        self.set_priors()

    def go(self):
        """
        loop over objects, load data, create the round
        info, copy to the output
        """
        nobj = self.model_fits.size

        for i in xrange(nobj):
            print("%d:%d" % (i+1,nobj))
            self.process_obj(i)

    def get_data(self):
        """
        get the round data struct
        """
        return self.data

    def process_obj(self, index):
        """
        process a single object
        """
        try:
            mbo=self.get_mb_obs_list(index)
            print("    ",mbo[0][0].image.shape)
            self.process_models(index,mbo)
            self.data['round_flags'][index]=0
        except MissingFit as e:
            print("   skipping object: %s" % str(e))
            self.data['round_flags'][index] = MISSING_FIT

    def process_models(self, index, mbo):

        for model in self['model_pars']:
            print("    model:",model)

            try:
                self.process_model(index, mbo, model)
            except GMixRangeError as e:
                print("   bad model found: %s" % str(e))
                n=Namer(model)
                self.data[n('round_flags')][index] = BAD_MODEL

    def process_model(self, index, mbo_orig, model):
        """
        calc round stats for the requested model
        """
        mbo_round, pars_round = self.get_round_mbo(index, mbo_orig, model)

        n=Namer(model)
        pname=n('pars')
        #pname=n('pars_best')
        pars=self.model_fits[n('pars')][index].copy()

        if self['use_logpars']:
            pars[4:] = exp(pars[4:])

        s2n,s2n_flags=self.get_s2n_r(mbo_round)
        Ts2n,Ts2n_flags=self.get_Ts2n_r(mbo_round, model, pars_round)

        self.data[n('round_flags')][index] = s2n_flags | Ts2n_flags
        self.data[n('T_r')][index] = pars_round[4]
        self.data[n('s2n_r')][index] = s2n
        self.data[n('T_s2n_r')][index] = Ts2n

        self.print_one(self.data[index],n)

    def get_s2n_r(self, mbo):
        """
        input must be the round version of the observation
        """
        s2n_sum=0.0
        for obslist in mbo:
            for obs in obslist:
                gmix=obs.gmix
                s2n_sum += gmix.get_model_s2n_sum(obs)

        if s2n_sum < 0.0:
            s2n_sum=0.0
        s2n=sqrt(s2n_sum)

        flags=0
        return s2n, flags

    def get_Ts2n_r(self, mbo, model, pars_round):
        """
        input round version of observations
        """

        Ts2n=-9999.0
        flags=0

        fitter=ngmix.fitting.LMSimple(mbo, model,
                                      prior=self.priors[model],
                                      use_logpars=self['use_logpars'])

        try:
            fitter._setup_data(pars_round)

            try:
                cov=fitter.get_cov(pars_round, 1.0e-3, 5.0)
                if cov[4,4] > 0:
                    Ts2n=pars_round[4]/sqrt(cov[4,4])
                else:
                    flags = TS2N_FAIL
            except LinAlgError:
                print("    failure: could not invert Hess in Ts2n")
                flags = TS2N_FAIL

        except GMixRangeError as e:
            print("    failure: bad model in Ts2n calc")
            flags = TS2N_FAIL

        return Ts2n, flags



    def get_round_mbo(self, index, mbo, model):
        """
        get round observations, with simulated
        galaxy images
        """
        new_mbo = MultiBandObsList()

        bpars=zeros(6)

        n=Namer(model)
        pname=n('pars')
        #pname=n('pars_best')
        pars=self.model_fits[n('pars')][index].copy()

        if self['use_logpars']:
            pars[4:] = exp(pars[4:])

        for band in xrange(self.nband):
            bpars[0:5] = pars[0:5] 
            bpars[5] = pars[5+band]
            obslist=mbo[band]

            new_obslist = self.get_round_band_obslist(bpars,model,obslist)

            new_mbo.append( new_obslist )

        factor=ngmix.shape.get_round_factor(pars[2], pars[3])
        Tround = pars[4]*factor

        pars_round=pars.copy()
        pars_round[2]=0.0
        pars_round[3]=0.0
        pars_round[4]=Tround

        return new_mbo, pars_round

    def get_round_band_obslist(self, pars, model, obslist):
        """
        get round version of obslist, with simulated galaxy image
        """
        new_obslist=ObsList()

        gm0=ngmix.GMixModel(pars, model)

        gm0round=gm0.make_round()

        s2n_sum=0.0
        for obs in obslist:
            new_obs = self.get_round_obs(obs, gm0round)
            new_obslist.append( new_obs )

        return new_obslist

    def get_round_obs(self, obs, gm0round):
        """
        get round version of obs, with simulated galaxy image
        """
        psf_round=obs.psf.gmix.make_round()

        gm_round = gm0round.convolve( psf_round )
        im_nonoise=gm_round.make_image(obs.image.shape,
                                       jacobian=obs.jacobian)
        
        noise=1.0/sqrt( median(obs.weight) )
        nim = numpy.random.normal(scale=noise, size=im_nonoise.shape)
        im = im_nonoise + nim

        psf_obs=Observation(zeros( (1,1) ), gmix=psf_round)

        new_obs=Observation(im,
                            gmix=gm_round,
                            weight=obs.weight,
                            jacobian=obs.jacobian,
                            psf=psf_obs)

        new_obs.im_nonoise=im
        return new_obs

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

    def print_one(self, d, n):
        mess="    s2n: %.1f s2n_r: %.1f Ts2n: %.2f Ts2n_r: %.2f"
        tup=(d[n('s2n_w')],
             d[n('s2n_r')],
             d[n('T_s2n')],
             d[n('T_s2n_r')])
        print(mess % tup)


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

    def set_priors(self):

        priors={}
        for model in self['model_pars']:
            mpars=self['model_pars'][model]
            assert mpars['g_prior_type']=="cosmos-sersic","g prior must be cosmos-sersic"
            assert mpars['T_prior_type']=="TwoSidedErf","T prior must be TwoSidedErf"
            assert mpars['counts_prior_type']=="TwoSidedErf","counts prior must be TwoSidedErf"
            assert mpars['cen_prior_type']=="dgauss","cen prior must be dgauss"

            width=mpars['cen_prior_pars'][0]
            cen_prior=ngmix.priors.CenPrior(0.0, 0.0, width, width)

            g_prior = ngmix.priors.make_gprior_cosmos_sersic(type='erf')

            Tpars=mpars['T_prior_pars']
            T_prior=ngmix.priors.TwoSidedErf(*Tpars)

            countspars=mpars['counts_prior_pars']
            counts_prior=ngmix.priors.TwoSidedErf(*countspars)

            counts_priors = [counts_prior]*self.nband
            prior = ngmix.joint_prior.PriorSimpleSep(cen_prior,
                                                     g_prior,
                                                     T_prior,
                                                     counts_priors)
            priors[model]=prior

        self.priors=priors



    def get_dtype(self):
        dt=[('id','i8'),
            ('number','i4'),
            ('round_flags','i4')]

        for model in self['model_pars']:
            n=Namer(model)

            dt += [(n('round_flags'),'i4'),
                   (n('T'),'f8'),
                   (n('s2n_w'),'f8'),
                   (n('T_s2n'),'f8'),
                   (n('T_r'),'f8'),
                   (n('s2n_r'),'f8'),
                   (n('T_s2n_r'),'f8')]

        return dt

    def set_struct(self):
        dt=self.get_dtype()

        nobj=self.model_fits.size
        st=zeros(nobj, dtype=dt)

        st['id'] = self.model_fits['id']
        st['number'] = self.model_fits['number']
        st['round_flags'] = NO_ATTEMPT

        for model in self['model_pars']:
            n=Namer(model)

            st[n('round_flags')] = NO_ATTEMPT

            st[n('T')]=self.model_fits[n('T')]
            st[n('s2n_w')]=self.model_fits[n('s2n_w')]
            st[n('T_s2n')]=self.model_fits[n('T_s2n')]

            st[n('T_r')]=DEFVAL
            st[n('s2n_r')]=DEFVAL
            st[n('T_s2n_r')]=DEFVAL

        self.data=st
