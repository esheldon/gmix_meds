
def clip_element_wise(arr, minvals, maxvals):
    """
    min vals are 5 element, maxvals is all
    """
    for i in xrange(arr.size):
        arr[i] = arr[i].clip(min=minvals[i],max=maxvals[i])


class UtterFailure(Exception):
    """
    could not make a good guess
    """
    def __init__(self, value):
         self.value = value
    def __str__(self):
        return repr(self.value)


class Namer(object):
    def __init__(self, front=None):
        self.front=front
    def __call__(self, name):
        if self.front is None or self.front=='':
            return name
        else:
            return '%s_%s' % (self.front, name)

from ngmix import print_pars

class GuesserBase(object):
    def _fix_guess(self, guess, prior, ntry=4):
        from ngmix.priors import LOWVAL

        #guess[:,2]=-9999
        n=guess.shape[0]
        for j in xrange(n):
            for itry in xrange(ntry):
                try:
                    lnp=prior.get_lnprob_scalar(guess[j,:])

                    if lnp <= LOWVAL:
                        dosample=True
                    else:
                        dosample=False
                except GMixRangeError as err:
                    dosample=True

                if dosample:
                    print_pars(guess[j,:], front="bad guess:")
                    if itry < ntry:
                        guess[j,:] = prior.sample()
                    else:
                        raise UtterFailure("could not find a good guess")
                else:
                    break


class FromMCMCGuesser(GuesserBase):
    """
    get guesses from a set of trials
    """
    def __init__(self, trials, sigmas):
        self.trials=trials
        self.sigmas=sigmas
        self.npars=trials.shape[1]

        #self.lnprobs=lnprobs
        #self.lnp_sort=lnprobs.argsort()

    def __call__(self, n=None, get_sigmas=False, prior=None):
        """
        get a random sample from the best points
        """
        import random

        if n is None:
            is_scalar=True
            n=1
        else:
            is_scalar=False

        # choose randomly from best
        #indices = self.lnp_sort[-n:]
        #guess = self.trials[indices, :]

        trials=self.trials
        np = trials.shape[0]

        rand_int = random.sample(xrange(np), n)
        guess=trials[rand_int, :]

        if prior is not None:
            self._fix_guess(guess, prior)

        w,=numpy.where(guess[:,4] <= 0.0)
        if w.size > 0:
            guess[w,4] = 0.05*srandu(w.size)

        for i in xrange(5, self.npars):
            w,=numpy.where(guess[:,i] <= 0.0)
            if w.size > 0:
                guess[w,i] = (1.0 + 0.1*srandu(w.size))

        #print("guess from mcmc:")
        #for i in xrange(n):
        #    print_pars(guess[i,:], front="%d: " % i)

        if is_scalar:
            guess=guess[0,:]

        if get_sigmas:
            return guess, self.sigmas
        else:
            return guess


class FromPSFGuesser(GuesserBase):
    """
    get full guesses from just T,fluxes associated with
    psf

    should make this take log values...
    """
    def __init__(self, T, fluxes, scaling='linear'):
        self.T=T
        self.fluxes=fluxes
        self.scaling=scaling

        self.log_T = numpy.log10(T)
        self.log_fluxes = numpy.log10(fluxes)

    def __call__(self, n=1, prior=None, **keys):
        """
        center, shape are just distributed around zero
        """
        fluxes=self.fluxes
        nband=fluxes.size
        np = 5+nband

        guess=numpy.zeros( (n, np) )
        guess[:,0] = 0.01*srandu(n)
        guess[:,1] = 0.01*srandu(n)
        guess[:,2] = 0.1*srandu(n)
        guess[:,3] = 0.1*srandu(n)
        #guess[:,4] = numpy.log10( self.T*(1.0 + 0.2*srandu(n)) )

        if self.scaling=='linear':
            if self.T <= 0.0:
                guess[:,4] = 0.05*srandu(n)
            else:
                guess[:,4] = self.T*(1.0 + 0.1*srandu(n))

            fluxes=self.fluxes
            for band in xrange(nband):
                if fluxes[band] <= 0.0:
                    guess[:,5+band] = (1.0 + 0.1*srandu(n))
                else:
                    guess[:,5+band] = fluxes[band]*(1.0 + 0.1*srandu(n))

        else:
            guess[:,4] = self.log_T + 0.1*srandu(n)

            for band in xrange(nband):
                guess[:,5+band] = self.log_fluxes[band] + 0.1*srandu(n)

        if prior is not None:
            self._fix_guess(guess, prior)

        if n==1:
            guess=guess[0,:]
        return guess


class FixedParsGuesser(GuesserBase):
    """
    just return a copy of the input pars
    """
    def __init__(self, pars, pars_err):
        self.pars=pars
        self.pars_err=pars_err

    def __call__(self, get_sigmas=False, prior=None):
        """
        center, shape are just distributed around zero
        """

        guess=self.pars.copy()
        if get_sigmas:
            return guess, self.pars_err
        else:
            return guess


class FromParsGuesser(GuesserBase):
    """
    get full guesses from just T,fluxes associated with
    psf
    """
    def __init__(self, pars, pars_err, scaling='linear'):
        self.pars=pars
        self.pars_err=pars_err
        self.scaling=scaling

    def __call__(self, n=None, get_sigmas=False, prior=None):
        """
        center, shape are just distributed around zero
        """

        if n is None:
            n=1
            is_scalar=True
        else:
            is_scalar=False

        pars=self.pars
        npars=pars.size

        width = pars*0 + 0.1

        guess=numpy.zeros( (n, npars) )

        guess[:,0] = width[0]*srandu(n)
        guess[:,1] = width[1]*srandu(n)

        guess_shape=get_shape_guess(pars[2],pars[3],n,width[2:2+2])
        guess[:,2]=guess_shape[:,0]
        guess[:,3]=guess_shape[:,1]

        for i in xrange(4,npars):
            if self.scaling=='linear':
                if pars[i] <= 0.0:
                    guess[:,i] = width[i]*srandu(n)
                else:
                    guess[:,i] = pars[i]*(1.0 + width[i]*srandu(n))
            else:
                # we add to log pars!
                guess[:,i] = pars[i] + width[i]*srandu(n)

        if prior is not None:
            self._fix_guess(guess, prior)

        if is_scalar:
            guess=guess[0,:]

        if get_sigmas:
            return guess, self.pars_err
        else:
            return guess


class FromAlmostFullParsGuesser(GuesserBase):
    """
    get full guesses from just g1,g2,T,fluxes associated with
    psf
    """
    def __init__(self, pars, pars_err, scaling='linear'):
        self.pars=pars
        self.pars_err=pars_err
        self.scaling=scaling

    def __call__(self, n=None, get_sigmas=False, prior=None):
        """
        center is just distributed around zero
        """

        if n is None:
            n=1
            is_scalar=True
        else:
            is_scalar=False

        pars=self.pars
        npars=pars.size

        width = pars*0 + 0.1

        guess=numpy.zeros( (n, npars) )

        guess[:f prior is not None:
            self._fix_guess(guess, prior)

        if is_scalar:
            guess=guess[0,:]

        if get_sigmas:
            return guess, self.pars_err
        else:
            return guess


class FromFullParsGuesser(GuesserBase):
    """
    get full guesses
    """
    def __init__(self, pars, pars_err, scaling='linear'):
        self.pars=pars
        self.pars_err=pars_err
        self.scaling=scaling

    def __call__(self, n=None, get_sigmas=False, prior=None):
        if n is None:
            n=1
            is_scalar=True
        else:
            is_scalar=False

        pars=self.pars
        npars=pars.size

        width = pars*0 + 0.1

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


def get_shape_guess(g1, g2, n, width):
    """
    Get guess, making sure in range
    """

    guess=numpy.zeros( (n, 2) )
    shape=ngmix.Shape(g1, g2)

    for i in xrange(n):

        while True:
            try:
                g1_offset = width[0]*srandu()
                g2_offset = width[1]*srandu()
                shape_new=shape.copy()
                shape_new.shear(g1_offset, g2_offset)
                break
            except GMixRangeError:
                pass

        guess[i,0] = shape_new.g1
        guess[i,1] = shape_new.g2

    return guess

