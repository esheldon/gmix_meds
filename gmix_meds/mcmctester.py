import numpy as np
from scipy.optimize import curve_fit

def _gfunct(x, lga, alpha, lgkp):
    z=lgkp-x   #x=lnj
    w=np.exp(alpha*z)
    return lga+np.log(w/(w+1.0))   #f=lnP    

class MCMCTester(object):
    """
    MCMCTester implements a spectral method to test MCMC chains.
    
    It has been translated to python from IDL by Matthew Becker. The 
    algorithm and code is due to Martin Bucher and Joanna Dunkley circa 2004. 
    See this paper on the arXiv (http://arxiv.org/pdf/astro-ph/0405462v1.pdf).
        
    Notes from Original Authors
    ---------------------------    
    Written by Martin Bucher and Joanna Dunkley 2004
    Convergence test as in Dunkley et al 2004. Calculates the log of the
    power spectrum of each parameter, ie the FFT^2 of f_data, stored as
    lnPk. 
    Fits lnPk with the fitting function gfunct with three variable
    parameters Po, alpha, jstar, and returns the fitted log power 
    spectra yfit_arr.
    Po is P(k=0), alpha the slope of the small scale behaviour and
    jstar the wavenumber where the spectrum turns over. r is the
    convergence ratio.
    
    Plots the raw and fitted power spectra for each parameter (set to
    plot P(k)/n_steps)
    Note: variable parameter jmax= range over which spectra are fitted. Should 
    be ~10*jstar for fitting well so may have to iterate and check by
    eye the spectra have been well fit.
    For convergence require that 
    (1) the power spectrum is flat at small k - check eg jstar>20
    (2) the sample mean variance is low: r<0.01 for each parameter 
    implies variance of mean < 1% variance of distribution.
    
    Parameters
    ----------
    chain : array-like
        the MCMC chain with dimensions (Nparameters,Nsamples)
    jmax : int (Default: 1000)
        maximum number of samples to use in k-space 
        must be less than Nsamples/2
    rmax : float (Default: 0.01)
        maximum allowed value of r for each parameter (see above)
    jstarmin : float (Default: 20.0)
        minimum value of jstar (see above)
    
    Methods
    -------
    
    __init__(chain=None) : init the class, optionally set the chain 
    
    set_chain(chain) : set the chain
    
    test_chain(jmax=1000,rmax=1e-2,jstarmin=20.0) : test the chain, returns True or False
    
    __call__() : alias to test_chain()
    
    plot() : makes plots of the chain power spectra w/ fits
    
    __repr__ : print out parameters from fits    
    
    """
    def __init__(self,chain=None):        
        if chain is not None:
            self.set_chain(chain)
    
    def set_chain(self,chain):
        cs = chain.shape
        assert len(cs) == 2,"Chain must have two dimensions!"
        assert cs[0] > cs[1],"Chain must have dimensions (Nsamples,Nparameters)!"
        
        self.chain = chain.copy()
        self.Ns = cs[0]
        self.Nd = cs[1]
        
        self.Po = np.zeros(self.Nd,dtype='f8')
        self.alpha = np.zeros(self.Nd,dtype='f8')
        self.jstar = np.zeros(self.Nd,dtype='f8')
        self.kstar = np.zeros(self.Nd,dtype='f8')
        self.r = np.zeros(self.Nd,dtype='f8')
        
        self.bias = 0.577216 #Euler -Mash constant, offset for E(lnP(k)) != lnP(K)
        
        # standardized chain
        self.chain_std = np.zeros_like(chain)

        self.means = self.chain.mean(axis=0)
        self.stds  = self.chain.std(axis=0)
        for i in xrange(self.Nd):
            self.chain_std[:,i] = (self.chain[:,i]-self.means[i])/self.stds[i]

        self.k = np.fft.fftfreq(self.Ns)*2.0*np.pi
        self.k = self.k[0:self.Ns/2]        
        self.lnPk = np.zeros((self.Nd,self.Ns/2))
        
        self.jmax = 1000
        if self.jmax > self.Ns/2:
            self.jmax = self.Ns/2
        
        self.chain_ok = False
            
    def test_chain(self,jmax=None,rmax=1e-2,jstarmin=20.0):
        self._do_test = False
        self.ffts = []
        self.ps = []
        
        if jmax != None:
            self.jmax = jmax
        assert self.jmax <= self.Ns/2, "jmax must be less than Nsamples/2!"                    
        
        #do FFTs, get powers and then do fits
        x = np.log(np.arange(self.jmax)+1.0)
        for i in xrange(self.Nd):
            #FFTs
            fft = np.fft.fft(self.chain_std[:,i])
            self.ffts.append(fft)
            
            #powers
            self.lnPk[i,:] = (np.log(1.0/self.Ns*np.abs(self.ffts[i])**2))[0:self.Ns/2]
            
            #the fit
            y = self.lnPk[i,0:self.jmax]
            p,pcov = curve_fit(_gfunct,x,y,p0=[2.0, 2.0, 3.0])
            self.ps.append(p.copy())
            
            self.Po[i] = np.exp(p[0]+self.bias)
            self.alpha[i] = p[1].copy()
            self.jstar[i] = np.exp(p[2])
            self.kstar[i] = 2.0*np.pi/self.Ns*self.jstar[i]
            self.r[i] = self.Po[i]/self.Ns
            
        return self._test_chain(rmax=rmax,jstarmin=jstarmin)

    def _test_chain(self,rmax=1e-2,jstarmin=20.0):
        if np.all(self.r < rmax) and np.all(self.jstar > jstarmin):
            self.chain_ok = True
            return True
        else:
            self.chain_ok = False
            return False        
            
    def __call__(self,jmax=None,rmax=1e-2,jstarmin=20.0):
        return self.test_chain(jmax=jmax,rmax=rmax,jstarmin=jstarmin)
        
    def plot(self):
        import matplotlib.pyplot as plt
        for i in xrange(self.Nd):
            p = self.ps[i]
            x = np.log(np.arange(self.jmax)+1.0)
            y = self.lnPk[i,0:self.jmax]
            
            plt.figure()
            plt.loglog(self.k[0:self.jmax],np.exp(y),'k-')            
            yfit = _gfunct(x,*p)+self.bias
            plt.loglog(self.k[0:self.jmax],np.exp(yfit),'r-')
            plt.xlabel(r'$k$')
            plt.xlabel(r'$P(k)$')
            plt.ylim(1e-5,1e3)
            plt.minorticks_on()
            plt.show()
            
    def __repr__(self):
        #print results
        base = "MCMCTester(chain(Nsamples=%d, Nparameters=%d))" % (self.Ns,self.Nd)
        if self._do_test == False:
            base += "\nPo         alpha      j*         r\n"
            for i in xrange(self.Nd):
                base += "%-10.4lg %-10.4lg %-10.4lg %-10.4lg\n" % (self.Po[i],self.alpha[i],self.jstar[i],self.r[i])
                
            base += '**For convergence, check j*>20 and r<0.01 for all parameters'
        
        return base
        

            
