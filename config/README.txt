ngmix012
    - new astrometry
    - more stable fitting I think; no gauss fit just emcee on the coadd, and
      base guess for me on that.
    - no box size cut
    - exp only?

010 was used in 008t after 007t with 011.  It is same but with psf guesses, trying
to reproduce results for 005t(008)

011 is now similar to 008 in that it is mh hybrid.  The difference is guessing
based on the coadd catalog parameters

test is currently as close to 008 as I can get except using emcee everywhere
and coadd


nmedstest3 - flat g prior, made no difference
