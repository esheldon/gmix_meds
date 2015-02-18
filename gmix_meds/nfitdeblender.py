from __future__ import print_function
import numpy
from .nfit import *
from ngmix import print_pars
import copy
from ngmix.fitting import LMSimple, MaxSimple, EIG_NOTFINITE
from ngmix.em import ApproxEMSimple

NOT_DEBLEND_MODEL = 2**29

#FIXME - add to util.py when merge is done
class FromFullParsErrGuesser(FromFullParsGuesser):
    """
    get full guesses
    """
    def __init__(self, pars, pars_err, scaling='linear', frac=0.1):
        self.pars=pars
        self.pars_err=pars_err
        self.scaling=scaling
        
        widths=frac*pars_err

        # remember that srandu has width 0.57
        sw=0.57
        widths[0:0+2] = widths[0:0+2].clip(min=1.0e-2, max=1.0)
        widths[2:2+2] = widths[2:2+2].clip(min=1.0e-2, max=1.0)
        widths[4:] = widths[4:].clip(min=1.0e-1, max=10.0)

        widths *= (1.0/sw)

        self.widths=widths

    def __call__(self, n=None, get_sigmas=False, prior=None):
        if n is None:
            n=1
            is_scalar=True
        else:
            is_scalar=False

        pars=self.pars
        npars=pars.size

        width = self.widths

        guess=numpy.zeros( (n, npars) )

        for j in xrange(n):
            itr = 0
            maxitr = 100
            while itr < maxitr:
                for i in xrange(npars):
                    if self.scaling=='linear' and i >= 4:
                        if pars[i] <= 0.0:
                            guess[j,:] = width[i]*srandu(1)
                        else:
                            guess[j,i] = pars[i]*(1.0 + width[i]*srandu(1))
                    else:
                        # we add to log pars!
                        guess[j,i] = pars[i] + width[i]*srandu(1)

                if numpy.abs(guess[j,2]) < 1.0 \
                        and numpy.abs(guess[j,3]) < 1.0 \
                        and guess[j,2]*guess[j,2] + guess[j,3]*guess[j,3] < 1.0:
                    break
                itr += 1

        if prior is not None:
            self._fix_guess(guess, prior)

        if is_scalar:
            guess=guess[0,:]

        if get_sigmas:
            return guess, self.pars_err
        else:
            return guess

class MLDeblender(MedsFit):
    """
    Maximum Likelihood Deblender
    """

    def _check_convergence(self,fofmems):
        """
        check convergence of fits
        """
        
        dlevel = 0
        
        npars = 5+self['nband']
        maxabs = numpy.zeros(npars,dtype='f8')
        maxabs[:] = -numpy.inf
        maxfrac = numpy.zeros(npars,dtype='f8')
        maxfrac[:] = -numpy.inf
        maxerr = numpy.zeros(npars,dtype='f8')
        maxerr[:] = -numpy.inf
        
        print('convergence:')
        for mindex in fofmems:
            if self['debug_level'] >= dlevel:
                print('    mindex:',mindex)
                print('    coadd_objects_id: %ld' % self.meds_list[0]['id'][mindex])
                print('    number: %ld' % self.meds_list[0]['number'][mindex])
            for model in self.fof_models_to_fit[mindex]:                
                old = self.prev_data[model+'_pars'][mindex]
                new = self.data[model+'_pars'][mindex]
                absdiff = numpy.abs(new-old)
                absfracdiff = numpy.abs(new/old-1.0)
                abserr = numpy.abs((old-new)/numpy.sqrt(numpy.diag(self.data[model+'_pars_cov'][mindex])))
                
                for i in xrange(npars):
                    if absdiff[i] > maxabs[i]:
                        maxabs[i] = copy.copy(absdiff[i])
                    if absfracdiff[i] > maxfrac[i]:
                        maxfrac[i] = copy.copy(absfracdiff[i])
                    if abserr[i] > maxerr[i]:
                        maxerr[i] = copy.copy(abserr[i])
                
                if self['debug_level'] >= dlevel:
                    print('        %s:' % model)
                    print_pars(old,        front='            old      ')
                    print_pars(new,        front='            new      ')
                    print_pars(absdiff,    front='            abs diff ')
                    print_pars(absfracdiff,front='            frac diff')
                    print_pars(abserr,front='            err diff ')
        
                    
        fmt = "%8.3g "*len(maxabs)
        print("    max abs diff : "+fmt % tuple(maxabs))
        print("    max frac diff: "+fmt % tuple(maxfrac))
        print("    max err diff : "+fmt % tuple(maxerr))
        
        self.maxabs = maxabs
        self.maxfrac = maxfrac
        self.maxerr = maxerr
        
        if numpy.all((maxabs <= self['deblend_maxabs_conv']) | (maxfrac <= self['deblend_maxfrac_conv']) | (maxerr <= self['deblend_maxerr_conv'])):
            return True
        else:
            return False

    def do_fits(self):
        """
        Fit all objects in our list
        """
        
        #fix up conv criterion
        for key in ['deblend_maxabs_conv','deblend_maxfrac_conv']:
            if len(self[key]) < self['nband']+5:
                ndiff = self['nband']+5 - len(self[key])
                for i in xrange(ndiff):
                    self[key].append(self[key][-1])
                self[key] = numpy.array(self[key])
                assert len(self[key]) == self['nband']+5
        
        t0=time.time()
        num = len(self.fofids)
        
        for fofid in self.fofids:
            if self['save_obs_per_fof']:
                self.fof_mb_obs_list = {}
                
            #################################
            #first get fiducial models
            #################################
            print('================================================================================')
            print('================================================================================')
            print('FoF id:',fofid)
            print('first fit:')
            
            fofmems = self.fofid2mindex[fofid]
            q = numpy.argsort(self.meds_list[0]['box_size'][fofmems])
            q = q[::-1]
            sfofmems = fofmems[q]
            
            self.fof_models_to_fit = {}
            for mindex in fofmems:
                self.fof_models_to_fit[mindex] = copy.copy(self['fit_models'])
            
            loc = 0
            for mindex in sfofmems:
                if self.data['processed'][mindex] > 0:
                    continue
                
                print('index: %d:%d' % (mindex,len(fofmems)-1))
                print('    on % 5d of % 5d' % (loc+1,len(fofmems)))
                ti = time.time()
                self.fit_obj(mindex,model_neighbors=False)
                ti = time.time()-ti
                print('    time:',ti)
                
                loc += 1
                
            
            #################################
            #now deblend the light
            #################################
                
            #if only 1 mem, no deblending needed!
            if len(fofmems) == 1:
                continue
                        
            converged = False
            
            for itr in range(self['max_deblend_itr']):
                #copy data
                print (' ')
                print('================================================================================')
                print('================================================================================')
                print('deblend itr: %d of %d' % (itr+1,self['max_deblend_itr']))
                
                self.prev_data = self.data.copy()
                
                loc = 0
                rfofmems = numpy.random.permutation(fofmems)
                for mindex in rfofmems:
                    self.data['processed'][mindex] += 1
                    
                    print('index: %d:%d' % (mindex,len(rfofmems)-1), )
                    print('    on % 5d of % 5d' % (loc+1,len(fofmems)))
                    ti = time.time()                    
                    #skip if flags are set from first try
                    if self.data['flags'][mindex] > 0:
                        print('    flags:',self.data['flags'][mindex])
                        print('    PASS')
                        pass
                    else:
                        self.fit_obj(mindex,model_neighbors=True)
                    ti = time.time()-ti
                    print('    time:',ti)
                    loc += 1
                    
                    
                if itr == self['deblend_iter_decide'] and len(self['fit_models']) > 1:
                    for mindex in fofmems:
                        chi2 = []
                        for model in self['fit_models']:
                            chi2.append(self.data[model+'_chi2per'][mindex])
                        chi2 = numpy.array(chi2)
                        q = numpy.argmin(chi2)
                        for model_ind,model in enumerate(self['fit_models']):
                            if model_ind != q:
                                self.data[model+'_flags'][mindex] |= NOT_DEBLEND_MODEL
                        self.fof_models_to_fit[mindex] = [self['fit_models'][q]]                    
                else:
                    converged = self._check_convergence(fofmems)
                    if converged:
                        break
                
                tm=time.time()-t0
                self._try_checkpoint(tm) # only at certain intervals
                    
            print('FoF id:',fofid)
            print('    converged:',converged)
            print('    itr:',itr+1)
            fmt = "%8.3g "*len(self.maxabs)
            print("    max abs diff : "+fmt % tuple(self.maxabs))
            print("    max frac diff: "+fmt % tuple(self.maxfrac))
            print("    max err diff : "+fmt % tuple(self.maxerr))
            
            if self['save_obs_per_fof']:
                del self.fof_mb_obs_list
        
        tm=time.time()-t0
        print("time:",tm)
        print("time per fof:",tm/num)
        
    def _fit_all_models(self):
        """
        Fit psf flux and other models
        """
        dindex=self.dindex

        if self.data['processed'][dindex] == 0:        
            flags=0
            # fit both coadd and se psf flux if exists
            self._fit_psf_flux()

            s2n=self.data['psf_flux'][dindex,:]/self.data['psf_flux_err'][dindex,:]
            max_s2n=numpy.nanmax(s2n)

            if max_s2n >= self['min_psf_s2n'] and len(self['fit_models']) > 0:
                fit_obj = True
            else:
                fit_obj = False
                mess="    psf s/n too low: %s (%s)" % (max_s2n,self['min_psf_s2n'])
                print(mess)            
                flags |= LOW_PSF_FLUX
        else:
            #we don't get here if flags != 0 from previous attempt
            #test is in do_fits above
            fit_obj = True
            flags = self.data['flags'][dindex]
            assert flags == 0
        
        if fit_obj:
            for model in self.fof_models_to_fit[dindex]:
                print('    fitting:',model)

                if self['fit_coadd_galaxy']:
                    print('    coadd')
                    self._run_model_fit(model, coadd=True)

                if self['fit_me_galaxy']:
                    print('    multi-epoch')
                    self._run_model_fit(model, coadd=False)

        return flags

    def _run_model_fit(self, model, coadd=False):
        """
        wrapper to run fit, copy pars, maybe make plots
        sets .fitter or .coadd_fitter
        """
        dindex = self.mindex
        
        if coadd:
            if self.data['processed'][dindex] == 0 or self.data['coadd_'+model+'_flags'][dindex] > 0:
                self.guesser = self._get_guesser('coadd_psf')
            else:
                pars = self.data['coadd_'+model+'_pars'][dindex]
                errs = numpy.sqrt(numpy.diag(self.data['coadd_'+model+'_pars_cov'][dindex]))
                self.guesser = FromFullParsErrGuesser(pars,errs,frac=self['deblend_guess_frac'])
            mb_obs_list=self.sdata['coadd_mb_obs_list']
        else:
            if self.data['processed'][dindex] == 0 or self.data[model+'_flags'][dindex] > 0:
                self.guesser = self._get_guesser('me_psf')
            else:
                pars = self.data[model+'_pars'][dindex]
                errs = numpy.sqrt(numpy.diag(self.data[model+'_pars_cov'][dindex]))
                self.guesser = FromFullParsErrGuesser(pars,errs,frac=self['deblend_guess_frac'])
            mb_obs_list=self.sdata['mb_obs_list']

        fitter=self._fit_model(mb_obs_list, model)

        self._copy_simple_pars(fitter, coadd=coadd)

        self._print_res(fitter, coadd=coadd)

        if self['make_plots']:
            self._do_make_plots(fitter, model, coadd=coadd)

        if coadd:
            self.coadd_fitter=fitter
        else:
            self.fitter=fitter
            
    def _fix_fitter(self, fitter):
        #wow this next bit is a total hack...
        if not hasattr(fitter,'get_best_pars'):
            fitter.get_best_pars = lambda : fitter.get_result()['pars']
        
        if self['deblend_force_flags'] and fitter._result['flags'] > 0:
            fitter._result['flags'] = 0
            
        Np = len(fitter._result['pars'])
        if 'pars_cov' not in fitter._result:
            cov = numpy.zeros((Np,Np),dtype=float)
            fitter._result['pars_cov'] = cov
            
        if 'pars_err' not in fitter._result:
            fitter._result['pars_err'] = numpy.sqrt(numpy.diag(fitter._result['pars_cov']))
                
        if 'g_cov' not in fitter._result:
            fitter._result['g_cov'] = fitter._result['pars_cov'][2:2+2, 2:2+2]
                
        return fitter
            
    def _fit_model(self, mb_obs_list, model):
        """
        Fit all the simple models
        """
        
        if self.data['processed'][self.mindex] == 0:
            fitter = self._fit_simple_max(mb_obs_list, model, self['max_pars'])
        else:
            fitter = self._fit_simple_lm(mb_obs_list, model, self)
        
        res = fitter.get_result()
        print_pars(res['pars'],front='        final:')
        print('        flags:',res['flags'])
        
        fitter = self._fix_fitter(fitter)
            
        #FIXME - not doing MCMC stats clearly - any side effects?
        # also adds .weights attribute
        #self._calc_mcmc_stats(fitter, model)

        return fitter
        
    def _fit_simple_max(self, mb_obs_list, model, max_pars):
        """
        parameters
        ----------
        mb_obs_list: MultiBandObsList
            The observations to fit
        model: string
            model to fit
        params: dict
            from the config file
        """

        prior=self['model_pars'][model]['gflat_prior']

        fitter=MaxSimple(mb_obs_list,
                         model,
                         prior=prior,
                         method=max_pars['method'])

        guess=self.guesser(prior=prior)
        print_pars(guess,front='        guess:')
        
        for i in xrange(max_pars['ntry']):
            fitter.run_max(guess, **max_pars)
            res=fitter.get_result()

            if (max_pars['ntry'] > 1) and (res['flags'] & EIG_NOTFINITE) != 0:
                # bad covariance matrix, need to get a new guess
                print_pars(res['pars'],front='        bad cov at pars')
                self.guesser=FromParsGuesser(guess, guess*0.1)
                guess=self.guesser(prior=prior)
                print_pars(guess,front='        new guess')
            else:
                break
            
        return fitter

    def _fit_simple_em(self, mb_obs_list, model, em_pars):
        """
        parameters
        ----------
        mb_obs_list: MultiBandObsList
            The observations to fit
        model: string
            model to fit
        params: dict
            from the config file
        """
        
        guess=self.guesser(prior=None)
        print_pars(guess,front='        guess:')                
        em_pars['guess'] = guess
        em_pars['verbose'] = True
        
        fitter=ApproxEMSimple(mb_obs_list,
                              model)
        
        fitter.run_em(**em_pars)
        fitter._result['flags'] = 0
        fitter._result['model'] = model
        
        return fitter

    def _fit_simple_lm(self, mb_obs_list, model, params, use_prior=True):
        """
        Fit one of the "simple" models, e.g. exp or dev
        use flat g prior
        """

        from ngmix.fitting import LMSimple

        if use_prior:
            prior=self['model_pars'][model]['gflat_prior']
        else:
            prior=None

        ok=False
        ntry=params['lm_ntry']
        for i in xrange(ntry):
            guess=self.guesser(prior=prior)
            print_pars(guess,front='        guess:')
            
            fitter=LMSimple(mb_obs_list,
                            model,
                            prior=prior,
                            lm_pars=params['lm_pars'])

            fitter.run_lm(guess)
            res=fitter.get_result()
            if res['flags']==0:
                ok=True
                break

        res['ntry']=i+1
        fitter._result['model'] = model
        
        return fitter

    
