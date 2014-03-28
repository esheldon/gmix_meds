#import numpy
#from numpy import sqrt

#from .lmfit import MedsFit, get_model_names, \
#        get_psf_ngauss, add_noise_matched, sigma_clip, \
#        _stat_names
from .lmfit import *
from .lmfit import _stat_names

try:
    from gmix_image.gmix_mcmc import MixMCSimple
except:
    print 'could not import gmix_image.gmix_mcmc'

class MedsMCMC(MedsFit):
    def __init__(self, meds_file, g_prior, **keys):

        super(MedsMCMC,self).__init__(meds_file, **keys)

        self.g_prior=g_prior
        self.nwalkers=keys.get('nwalkers',20)
        self.burnin=keys.get('burnin',400)
        self.nstep=keys.get('nstep',200)
        self.do_pqr=keys.get("do_pqr",False)
        self.mca_a=keys.get('mca_a',2.0)

        self.match_nwalkers=keys.get('match_nwalkers',20)
        self.match_burnin=keys.get('match_burnin',200)
        self.match_nstep=keys.get('match_nstep',200)

        self.when_prior=keys.get('when_prior','during')
        self.draw_gprior = keys.get('draw_gprior',True)

        self.cen_width = keys.get('cen_width',1.0)

        self.make_plots=keys.get('make_plots',False)
        self.prompt=keys.get('prompt',True)

    '''
    def _fit_obj(self, index):
        """
        Process the indicated object

        The first cutout is always the coadd, followed by
        the SE images which will be fit simultaneously
        """

        t0=time.time()

        self.data['flags'][index] = self._obj_check(index)
        if self.data['flags'][index] != 0:
            return 0

        imlist0,self.coadd = self._get_imlist(index)
        wtlist0=self._get_wtlist(index)
        jacob_list0=self._get_jacobian_list(index)

        self.data['nimage_tot'][index] = len(imlist0)
        print >>stderr,imlist0[0].shape
    
        keep_list,psf_gmix_list=self._fit_psfs(index,jacob_list0)
        if len(psf_gmix_list)==0:
            self.data['flags'][index] |= PSF_FIT_FAILURE
            return

        keep_list,psf_gmix_list=self._remove_bad_psfs(keep_list,psf_gmix_list)
        if len(psf_gmix_list)==0:
            self.data['flags'][index] |= PSF_LARGE_OFFSETS
            return

        imlist = [imlist0[i] for i in keep_list]
        wtlist = [wtlist0[i] for i in keep_list]
        jacob_list = [jacob_list0[i] for i in keep_list]
        self.data['nimage_use'][index] = len(imlist)

        sdata={'imlist':imlist,
               'wtlist':wtlist,
               'jacob_list':jacob_list,
               'psf_gmix_list':psf_gmix_list}

        if 'psf' in self.conf['fit_types']:
            self._fit_psf_flux(index, sdata)
        if 'simple' in self.conf['fit_types']:
            self._fit_simple_models(index, sdata)
        if 'cmodel' in self.conf['fit_types']:
            self._fit_cmodel(index, sdata)
        if 'match' in self.conf['fit_types']:
            self._fit_match(index, sdata)

        if self.debug >= 3:
            self._debug_image(sdata['imlist'][0],sdata['wtlist'][-1])

        self.data['time'][index] = time.time()-t0
    '''

    def _fit_simple_models(self, index, sdata):
        """
        Fit all the simple models
        """
        if self.debug:
            bsize=self.meds['box_size'][index]
            bstr='[%d,%d]' % (bsize,bsize)
            print >>stderr,'\tfitting simple models %s' % bstr

        psf_s2n=self.data['psf_flux'][index]/self.data['psf_flux_err'][index]
        make_plots=False

        for model in self.simple_models:
            if False:
                if 50 < psf_s2n < 75:
                    make_plots=True
                else:
                    make_plots=False
                    continue

            if (make_plots or self.make_plots) and self.prompt:
                self._show_coadd()

            print >>stderr,'    fitting:',model

            gm=self._fit_simple(index, model, sdata, make_plots=make_plots)
            res=gm.get_result()

            n=get_model_names(model)

            if self.debug:
                self._print_simple_stats(n, res)

            self._copy_simple_pars(index, res, n)

    def _fit_simple(self, index, model, sdata, make_plots=False):
        """
        Fit one of the "simple" models, e.g. exp or dev
        """
        if self.data['psf_flags'][index]==0:
            counts_guess=self.data['psf_flux'][index]
        else:
            # will come from median of input images
            counts_guess=None

        nwalkers,burnin,nstep=self._get_mcmc_pars(index)

        cen_guess=[0.0, 0.0]
        #sigma_guess=2.0/2.35 # FWHM of 2''
        #T_guess=2*sigma_guess**2
        T_guess=16.0
        gm=MixMCSimple(sdata['imlist'],
                       sdata['wtlist'],
                       sdata['psf_gmix_list'],
                       self.g_prior,
                       T_guess,
                       counts_guess,
                       cen_guess,
                       model,
                       jacob=sdata['jacob_list'],
                       cen_width=self.cen_width,
                       nwalkers=nwalkers,
                       burnin=burnin,
                       nstep=nstep,
                       mca_a=self.mca_a,
                       do_pqr=self.do_pqr,
                       when_prior=self.when_prior,
                       draw_gprior=self.draw_gprior,
                       prompt=self.prompt,
                       make_plots=make_plots or self.make_plots)
        if hasattr(gm,'tab'):
            imname='%s-dist-%05d.png' % (model,index)
            print >>stderr,imname
            gm.tab.write_img(800,800,imname)
        return gm

    def _get_mcmc_pars(self, index):
        nwalkers=self.nwalkers
        burnin=self.burnin
        nstep=self.nstep

        #return nwalkers,burnin,nstep

        if self.data['psf_flags'][index]==0:
            from math import ceil
            psf_s2n=self.data['psf_flux'][index]/self.data['psf_flux_err'][index]
            thresh=30.0
            if psf_s2n > thresh:
                nwalkers += int( (psf_s2n-thresh)**0.85 )
                if (nwalkers % 2) != 0:
                    nwalkers += 1
                if nwalkers > 100:
                    nwalkers=100
                print '    psf s/n:',psf_s2n,'nwalkers:',nwalkers
        else:
            nwalkers *= 2

        return nwalkers, burnin, nstep

    def _copy_simple_pars(self, index, res, n):

        self.data[n['flags']][index] = res['flags']

        if res['flags'] == 0:
            self.data[n['pars']][index,:] = res['pars']
            self.data[n['pars_cov']][index,:,:] = res['pcov']

            flux=res['pars'][5]
            flux_err=sqrt(res['pcov'][5,5])
            self.data[n['flux']][index] = flux
            self.data[n['flux_err']][index] = flux_err

            self.data[n['g']][index,:] = res['pars'][2:2+2]
            self.data[n['g_cov']][index,:,:] = res['pcov'][2:2+2,2:2+2]

            self.data[n['g_sens']][index,:] = res['gsens']
            if self.do_pqr:
                self.data[n['P']][index] = res['P']
                self.data[n['Q']][index,:] = res['Q']
                self.data[n['R']][index,:,:] = res['R']

            for sn in _stat_names:
                self.data[n[sn]][index] = res[sn]
        else:
            if self.debug:
                print >>stderr,'flags != 0, errmsg:',res['errmsg']
            if self.debug > 1 and self.debug < 3:
                self._debug_image(sdata['imlist'][0],sdata['wtlist'][0])


    def _fit_match(self, index, sdata, make_plots=False):
        chi2per=PDEFVAL
        flux=DEFVAL
        flux_err=PDEFVAL
        bres={'flags':0,
              'flux':DEFVAL,'flux_err':PDEFVAL,
              'chi2per':PDEFVAL, 'loglike':DEFVAL,
              'model':'nil'}


        if self.det_cat is None:
            bres0=self._get_best_simple_pars(self.data,index)
            if bres0['flags']==0:
                bres.update(bres0)
                bres['flux']=bres['pars'][5]
                bres['flux_err']=sqrt(bres['pcov'][5,5])
                bres['model'] = bres0['model']

            else:
                bres['flags']=bres0['flags']
        else:
            print >>stderr,"    fitting: match",
            bres0=self._get_best_simple_pars(self.det_cat,index)
            # if flags != 0 it is because we could not find a good fit of any
            # model
            if bres0['flags']==0:

                bres['model'] = bres0['model']
                mod=bres0['model']
                pars0=bres0['pars']
                if self.match_use_band_center:
                    pars0=self._set_center_from_band(index,pars0,mod)

                match_gmix = gmix_image.GMix(pars0, type=mod)
                start_counts=self._get_match_start(index, mod, match_gmix)
                match_gmix.set_psum(start_counts)

                if False:
                    psf_s2n=self.data['psf_flux'][index]/self.data['psf_flux_err'][index]
                    if psf_s2n < 100 and psf_s2n > 50:
                        make_plots=True
                    else:
                        make_plots=False


                gm=gmix_image.gmix_mcmc.MixMCMatch(sdata['imlist'],
                                                   sdata['wtlist'],
                                                   sdata['psf_gmix_list'],
                                                   match_gmix,
                                                   jacob=sdata['jacob_list'],
                                                   nwalkers=self.match_nwalkers,
                                                   burnin=self.match_burnin,
                                                   nstep=self.match_nstep,
                                                   mca_a=3,
                                                   prompt=self.prompt,
                                                   make_plots=make_plots or self.make_plots)

                if hasattr(gm,'tab'):
                    imname='match-dist-%05d.png' % index
                    print >>stderr,imname
                    gm.tab.write_img(800,800,imname)

                res=gm.get_result()
                flags=res['flags']
                if flags==0:
                    mess="  flux: %g +/- %g match_flux: %g +/- %g"
                    mess=mess % (pars0[5],sqrt(bres0['pcov'][5,5]), res['Flux'],res['Ferr'])
                    print >>stderr,mess
                    bres['flux']=res['Flux']
                    bres['flux_err']=res['Ferr']
                    bres['chi2per']=res['chi2per']
                    bres['loglike'] = res['loglike']
                else:
                    print >>stderr,"failure"
                    bres['flags']=flags

            else:
                bres['flags']=bres0['flags']

        self.data['match_flags'][index] = bres['flags']
        self.data['match_model'][index] = bres['model']
        self.data['match_chi2per'][index] = bres['chi2per']
        self.data['match_loglike'][index] = bres['loglike']
        self.data['match_flux'][index] = bres['flux']
        self.data['match_flux_err'][index] = bres['flux_err']
        if self.debug:
            fmt='\t\t%s[%s]: %g +/- %g'
            print >>stderr,fmt % ('match_flux',mod,bres['flux'],bres['flux_err'])



    def _print_simple_stats(self, ndict, res):                        
        fmt='\t\t%s: %g +/- %g'
        n=ndict
        if res['flags']==0:
            nm=n['flux']
            flux=res['pars'][5]
            flux_err=sqrt(res['pcov'][5,5])
            print >>stderr,fmt % (nm,flux,flux_err)



    def _make_struct(self):
        nobj=self.meds.size

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

            dt+=[(n['flags'],'i4'),
                 (n['pars'],'f8',simple_npars),
                 (n['pars_cov'],'f8',(simple_npars,simple_npars)),
                 (n['flux'],'f8'),
                 (n['flux_err'],'f8'),
                 (n['g'],'f8',2),
                 (n['g_cov'],'f8',(2,2)),
                 (n['g_sens'],'f8',2),
                 (n['P'],'f8'),
                 (n['Q'],'f8',2),
                 (n['R'],'f8',(2,2)),
                
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
              ('match_chi2per','f8'),
              ('match_loglike','f8'),
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
        data['match_chi2per'] = PDEFVAL
        data['match_loglike'] = DEFVAL
        data['match_model'] = 'nil'


        for model in simple_models:
            n=get_model_names(model)

            data[n['flags']] = NO_ATTEMPT

            data[n['pars']] = DEFVAL
            data[n['pars_cov']] = PDEFVAL
            data[n['flux']] = DEFVAL
            data[n['flux_err']] = PDEFVAL
            data[n['g']] = DEFVAL
            data[n['g_cov']] = PDEFVAL
            data[n['g_sens']] = DEFVAL
            data[n['P']] = DEFVAL
            data[n['Q']] = DEFVAL
            data[n['R']] = DEFVAL

            data[n['s2n_w']] = DEFVAL
            data[n['loglike']] = BIG_DEFVAL
            data[n['chi2per']] = PDEFVAL
            data[n['aic']] = BIG_PDEFVAL
            data[n['bic']] = BIG_PDEFVAL
        
        self.data=data



