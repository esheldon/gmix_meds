ngmix007t - nmeds011
ngmix008t - nmeds010

- ngmixtest2
    - coadd only, nmedstest2 config
* ngmixtest
    - using nmedstest.  This will be a run with multi-epoch mh
- ngmixtest3
    - using nmedstest2.  same as ngmixtest2 but r,i,z
    - the idea here is to see if using r,i,z will be closer to detmodel that
      fits to the g+r+i detection image to get the shape/profile.
    - no difference
- ngmixtest4
    - flat prior on g (nmedstest3), no difference

    - repurposing with T > 0 prior (as erf) to see if this is the problem.
        - no difference

    - repurposing for 2 gauss psf
