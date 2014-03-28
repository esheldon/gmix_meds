from sys import stderr
import os
import numpy
from numpy import sqrt
import fitsio
import meds

ZERO_WEIGHT_PIXELS=2**30


MAG_RANGE = [12.,25.]
SPREAD_RANGE =[-0.01,0.04]
STAR_MAG_RANGE = [15.5,19.0]
STAR_SPREAD_RANGE = [-0.002,0.002]

PIX_SCALE=0.265

WIDTHS = [0.25,0.50,0.75]

def collate_results(version, out_file):
    import glob
    d=get_tile_dir(version)

    flist=glob.glob(d+'/*models_%s.fits' % version)

    first=True
    print 'writing to:',out_file
    with fitsio.FITS(out_file,'rw',clobber=True) as fobj:

        for f in flist:
            print '    ',f
            data=fitsio.read(f)

            if first:
                fobj.write(data)
                first=False
            else:
                fobj[-1].append(data)

    print 'output in:',out_file

class StarPlotter(object):
    """
    Plot results from the collated file
    """
    def __init__(self, version, collated_file):
        print collated_file
        self.version=version
        self.data=fitsio.read(collated_file)

        self.offset_max=0.25

        self._extract_by_exp()

    def doplots(self, xfield='coadd_mag_auto', type='normal', show=False):
        import biggles
        biggles.configure('default','fontsize_min',1.5)
        nexp=len(self.expdict)
        i=1
        for expname,exp_objs in self.expdict.iteritems():
            print >>stderr,'%d/%d' % (i,nexp)

            if type=='em2_widths':
                self._process_exp_em2_widths(expname,exp_objs,xfield,show=show)
            elif type=='widths':
                self._process_exp_widths(expname,exp_objs,xfield,show=show)
            elif type=='normal':
                self._process_exp(expname,exp_objs,show=show)
            else:
                raise ValueError("bad type: '%s'" % type)
            i += 1


    def _process_exp_widths(self, expname, exp_objs, xfield, show=False):
        import biggles
        import pcolors
        import esutil as eu
        print >>stderr,'found',len(exp_objs),'ccds in',expname

        ccds = list(exp_objs.keys())
        nccd = len(ccds)

        plt=biggles.FramedArray(3,2)
        plt.title='%s  nccd: %d' % (expname,nccd)
        plt.yrange=[0.0,2.0]
        plt.uniform_limits=1

        if xfield=='coadd_mag_auto':
            plt.xlabel=r'$mag_{auto}$'
            plt.xrange=[15.4,19.2]
        elif xfield=='flux_max':
            plt.xlabel=r'$FLUX_MAX$'
            # need to fix
            plt.xrange=[0,40000]

        plt.ylabel=r'$2 \times sqrt(area/\pi) [arcsec]$'

        colors=pcolors.rainbow(nccd)

        fw75_slopes=numpy.zeros(nccd) -9999.e9
        fw50_slopes=numpy.zeros(nccd) -9999.e9
        fw25_slopes=numpy.zeros(nccd) -9999.e9

        for i,ccd in enumerate(ccds):
            wccd = numpy.array(exp_objs[ccd])
            color = colors[i]

            nw=len(wccd)

            # still cut just so we get the same objects
            w, = numpy.where(self.data['flags2'][wccd]==0
                              & (self.data['offset_arcsec'][wccd] < self.offset_max))
            if w.size != 0:
                w=wccd[w]


                self._add_to_plot(plt[0,0], xfield, 'fw75', w, color)
                cf=self._add_to_plot(plt[0,1], xfield, 'fw75', w, color, doslope=True)
                if cf is not None:
                    fw75_slopes[i] = cf[0]

                self._add_to_plot(plt[1,0], xfield, 'fw50', w, color)
                cf=self._add_to_plot(plt[1,1], xfield, 'fw50', w, color, doslope=True)
                if cf is not None:
                    fw50_slopes[i] = cf[0]

                self._add_to_plot(plt[2,0], xfield, 'fw25', w, color)
                cf=self._add_to_plot(plt[2,1], xfield, 'fw25', w, color, doslope=True)
                if cf is not None:
                    fw25_slopes[i] = cf[0]


        lab75=biggles.PlotLabel(0.1,0.9,'FW75',halign='left')
        lab50=biggles.PlotLabel(0.1,0.9,'FW50',halign='left')
        lab25=biggles.PlotLabel(0.1,0.9,'FW25',halign='left')
        plt[0,0].add(lab75)
        plt[1,0].add(lab50)
        plt[2,0].add(lab25)

        ws,=numpy.where(fw75_slopes > -1000)
        if ws.size > 2:
            self._add_slope(plt[0,1], fw75_slopes[ws])
        ws,=numpy.where(fw50_slopes > -1000)
        if ws.size > 2:
            self._add_slope(plt[1,1], fw50_slopes[ws])
        ws,=numpy.where(fw25_slopes > -1000)
        if ws.size > 2:
            self._add_slope(plt[2,1], fw25_slopes[ws])

        if show:
            plt.show()
            key=raw_input('hit a key (q to quit): ')
            if key.lower() == 'q':
                stop

        pngname=get_exp_size_mag_plot(self.version, expname, xfield, ext='png',type='widths')
        if not os.path.exists(os.path.dirname(pngname)):
            os.makedirs(os.path.dirname(pngname))
        print pngname
        plt.write_img(600,600,pngname)
        #plt.write_img(1100,1100,pngname)

    def _process_exp_em2_widths(self, expname, exp_objs,show=False):
        import biggles
        import pcolors
        import esutil as eu
        print >>stderr,'found',len(exp_objs),'ccds in',expname

        ccds = list(exp_objs.keys())
        nccd = len(ccds)

        plt=biggles.FramedArray(3,2)
        plt.title='%s  nccd: %d' % (expname,nccd)
        plt.yrange=[0.0,2.0]
        plt.xrange=[15.4,19.2]
        plt.uniform_limits=1

        plt.xlabel=r'$mag_{auto}$'
        plt.ylabel=r'$2 \times sqrt(area/\pi) [arcsec]$'

        colors=pcolors.rainbow(nccd)

        fw75_slopes=numpy.zeros(nccd) -9999.e9
        fw50_slopes=numpy.zeros(nccd) -9999.e9
        fw25_slopes=numpy.zeros(nccd) -9999.e9

        offset_arrlist=[]
        for i,ccd in enumerate(ccds):
            wccd = numpy.array(exp_objs[ccd])
            color = colors[i]

            nw=len(wccd)

            w, = numpy.where(self.data['flags2'][wccd]==0
                              & (self.data['offset_arcsec'][wccd] < self.offset_max))
            if w.size != 0:
                w=wccd[w]

                offset_arrlist.append( self.data['offset_arcsec'][w])

                self._add_to_plot(plt[0,0], xfield, 'fw75_em2', w, color)
                cf=self._add_to_plot(plt[0,1], xfield, 'fw75_em2', w, color, doslope=True)
                if cf is not None:
                    fw75_slopes[i] = cf[0]

                self._add_to_plot(plt[1,0], xfield, 'fw50_em2', w, color)
                cf=self._add_to_plot(plt[1,1], xfield, 'fw50_em2', w, color, doslope=True)
                if cf is not None:
                    fw50_slopes[i] = cf[0]

                self._add_to_plot(plt[2,0], xfield, 'fw25_em2', w, color)
                cf=self._add_to_plot(plt[2,1], xfield, 'fw25_em2', w, color, doslope=True)
                if cf is not None:
                    fw25_slopes[i] = cf[0]


        offsets=eu.numpy_util.combine_arrlist(offset_arrlist)
        mean_offset,soffset=eu.stat.sigma_clip(offsets)

        offlab = biggles.PlotLabel(0.9,0.9,'<offset>: %.3g' % mean_offset,halign='right')
        plt[0,0].add(offlab)

        lab75=biggles.PlotLabel(0.1,0.9,'FW75 EM2',halign='left')
        lab50=biggles.PlotLabel(0.1,0.9,'FW50 EM2',halign='left')
        lab25=biggles.PlotLabel(0.1,0.9,'FW20 EM2',halign='left')
        plt[0,0].add(lab75)
        plt[1,0].add(lab50)
        plt[2,0].add(lab25)

        ws,=numpy.where(fw75_slopes > -1000)
        if ws.size > 2:
            self._add_slope(plt[0,1], fw75_slopes[ws])
        ws,=numpy.where(fw50_slopes > -1000)
        if ws.size > 2:
            self._add_slope(plt[1,1], fw50_slopes[ws])
        ws,=numpy.where(fw25_slopes > -1000)
        if ws.size > 2:
            self._add_slope(plt[2,1], fw25_slopes[ws])

        if show:
            plt.show()
            key=raw_input('hit a key (q to quit): ')
            if key.lower() == 'q':
                stop

        pngname=get_exp_size_mag_plot(self.version, expname, xfield, ext='png',type='em2_widths')
        if not os.path.exists(os.path.dirname(pngname)):
            os.makedirs(os.path.dirname(pngname))
        print pngname
        plt.write_img(600,600,pngname)


    def _process_exp(self, expname, exp_objs,show=False):
        import biggles
        import pcolors
        print >>stderr,'found',len(exp_objs),'ccds in',expname
       
        ccds = list(exp_objs.keys())
        nccd = len(ccds)

        plt=biggles.FramedArray(3,2)
        plt.title='%s  nccd: %d' % (expname,nccd)
        plt.yrange=[0.7,1.6]
        plt.xrange=[15.4,19.2]
        plt.uniform_limits=1

        xlabel=r'$mag_{auto}$'
        ylabel=r'$FWHM [arcsec]$'

        plt.xlabel=xlabel
        plt.ylabel=ylabel


        colors=pcolors.rainbow(nccd)

        em1_slopes=numpy.zeros(nccd) -9999.e9
        em2_slopes=numpy.zeros(nccd) -9999.e9
        am_slopes=numpy.zeros(nccd) -9999.e9

        for i,ccd in enumerate(ccds):
            wccd = numpy.array(exp_objs[ccd])
            color = colors[i]

            nw=len(wccd)
            #print '    %d in ccd %d' % (nw,ccd)

            w1, = numpy.where(self.data['flags1'][wccd]==0)
            if w1.size != 0:
                w1=wccd[w1]
                self._add_to_plot(plt[0,0], 'em1', w1, color)
                cf=self._add_to_plot(plt[0,1], 'em1', w1, color,doslope=True)
                if cf is not None:
                    em1_slopes[i] = cf[0]

            offsets=self.data['offset_arcsec'][wccd]
            w2, = numpy.where(  (self.data['flags2'][wccd]==0)
                              & (offsets < self.offset_max))

            if w2.size != 0:
                w2=wccd[w2]
                self._add_to_plot(plt[1,0], 'em2', w2, color)
                cf=self._add_to_plot(plt[1,1], 'em2', w2, color,doslope=True)
                if cf is not None:
                    em2_slopes[i] = cf[0]

            wam, = numpy.where(self.data['amflags'][wccd]==0)
            if wam.size != 0:
                wam=wccd[wam]
                self._add_to_plot(plt[2,0], 'am', wam, color)
                cf=self._add_to_plot(plt[2,1], 'am', wam, color, doslope=True)
                if cf is not None:
                    am_slopes[i] = cf[0]

        lab1=biggles.PlotLabel(0.1,0.9,'ngauss: 1',halign='left')
        lab2_1=biggles.PlotLabel(0.1,0.9, "ngauss: 2", halign='left')
        lab2_2=biggles.PlotLabel(0.1,0.8,
                               "offset < %.2f''" % self.offset_max,
                               halign='left')
        amlab=biggles.PlotLabel(0.1,0.9, "AM", halign='left')
        plt[0,0].add(lab1)
        plt[1,0].add(lab2_1,lab2_2)
        plt[2,0].add(amlab)

        print >>stderr,'doing slopes'
        w,=numpy.where(em1_slopes > -1000)
        if w.size > 2:
            self._add_slope(plt[0,1], em1_slopes[w])
        w,=numpy.where(em2_slopes > -1000)
        if w.size > 2:
            self._add_slope(plt[1,1], em2_slopes[w])
        w,=numpy.where(am_slopes > -1000)
        if w.size > 2:
            self._add_slope(plt[2,1], am_slopes[w])

        if show:
            plt.show()
            key=raw_input('hit a key (q to quit): ')
            if key.lower() == 'q':
                stop

        pngname=get_exp_size_mag_plot(self.version, expname, xfield, ext='png')
        if not os.path.exists(os.path.dirname(pngname)):
            os.makedirs(os.path.dirname(pngname))
        print pngname
        plt.write_img(600,600,pngname)
        #epsname=get_exp_size_mag_plot(self.version, expname)
        #write_eps_and_convert(plt,epsname)


    def _add_slope(self, plt, slopes):
        import biggles
        from esutil.stat import sigma_clip
        slope,slope_sig=sigma_clip(slopes)
        slope_err=slope_sig/sqrt(slopes.size)

        #slope = slopes.mean()
        #slope_err = slopes.std()/sqrt(slopes.size)

        m='<slope>: %.2g +/- %.2g' % (slope,slope_err)
        slab = biggles.PlotLabel(0.9,0.9,m,halign='right')
        plt.add(slab)


    def _add_to_plot(self, plt, xfield, type, w, color, doslope=False):
        """

        Add widths to plot and plot a fit line. Very broad Sigma clipping is
        performed before fitting the line

        """
        import gmix_image
        import biggles
        from esutil.stat import sigma_clip


        if type=='fw25':
            width = self.data['fw25_arcsec'][w]
        elif type=='fw50':
            width = self.data['fw50_arcsec'][w]
        elif type=='fw75':
            width = self.data['fw75_arcsec'][w]
        elif type=='fw25_em2':
            width = self.data['fw25_arcsec_em2'][w]
        elif type=='fw50_em2':
            width = self.data['fw50_arcsec_em2'][w]
        elif type=='fw75_em2':
            width = self.data['fw75_arcsec_em2'][w]
        else:
            for i,wi in enumerate(w):

                width=numpy.zeros(w.size)
                if type=='em1':
                    gmix=gmix_image.GMix(self.data['pars1'][wi,:])
                elif type=='em2':
                    gmix=gmix_image.GMix(self.data['pars2'][wi,:])
                elif type=='am':
                    gmix=gmix_image.GMix(self.data['ampars'][wi,:])
                
                T=gmix.get_T()
                width[i] = 0.265*2.3548*sqrt(T/2.)

        x=self.data[xfield][w]
        pts=biggles.Points(x, width, type='dot', color=color)

        coeffs=None
        if doslope and w.size > 20:
            crap1,crap2,wsc=sigma_clip(width, nsig=5,get_indices=True)
            if wsc.size > 20:

                if xfield == 'flux_max':
                    w2,=numpy.where((x[wsc] > 5000) & (x[wsc] < 25000) )
                    xuse=x[wsc[w2]]
                    yuse=width[wsc[w2]]
                    #xuse=x[wsc]
                    #yuse=width[wsc]
                else:
                    xuse=x[wsc]
                    yuse=width[wsc]

                if xuse.size > 20:
                    coeffs=numpy.polyfit(xuse, yuse, 1)
                    ply=numpy.poly1d(coeffs)
                    yvals=ply(xuse)
                    cv=biggles.Curve(xuse, yvals, color=color)
                    plt.add(cv)

        plt.add(pts)

        return coeffs

    def _match_expname(self, expname):
        import esutil as eu
        logic=eu.numpy_util.strmatch(self.data['sename'],'.*'+expname+'.*')
        w,=numpy.where(logic)
        if w.size == 0:
            raise  ValueError("found zero '%s'?" % expname)
 
        return w

    def _extract_by_exp(self):
        print >>stderr,"extracting exposures"
        expdict={}
        for i,n in enumerate(self.data['sename']):

            ns = ( n.split('_') )
            expn='_'.join( ns[0:0+2] )

            end=ns[2]
            ccd=int( ( end.split('.') )[0] )

            if expn not in expdict:
                expdict[expn] = {ccd:[i]}
            else:
                if ccd not in expdict[expn]:
                    expdict[expn][ccd] = [i]
                else:
                    expdict[expn][ccd].append(i)

        self.expdict=expdict
        print >>stderr,'found',len(self.expdict),'exposures'

    def _extract_ccd(self, name):
        end=( name.split('_'))[2]
        ccd=int( ( end.split('.') )[0] )
        return ccd

class StarFitter(object):
    def __init__(self, version, meds_file):

        self.meds_file=meds_file
        self.version=version

        self._load_data()
        self._set_tileband()
        self._make_dirs()
        self._select_stars()

    def measure_stars(self):

        out_file=get_tile_path(self.version, self.tileband)
        print >>stderr,'opening output:',out_file

        first=True
        with fitsio.FITS(out_file,'rw',clobber=True) as fobj:
            for file_id in xrange(1,self.info.size):

                print >>stderr,self.bnames[file_id]

                slist,icutlist = self._find_stars_in_se(file_id)

                ns=len(slist)
                print >>stderr,'    found',ns,'stars'

                if ns > 0:
                    st=self._process_stars(file_id, slist, icutlist)
                    if first:
                        fobj.write(st,extname="model_fits")
                        first=False
                    else:
                        fobj[-1].append(st)
        print 'output is in:',out_file

    def _process_stars(self, file_id, slist, icutlist):
        import gmix_image
        ns=len(slist)
        st=self._get_struct(ns)
        st['tileband'] = self.tileband
        st['number'] = slist
        st['icutout'] = icutlist
        st['file_id'] = file_id
        st['sename']=self.bnames[file_id]
        st['coadd_mag_auto'] = self.cat['mag_auto'][slist]
        st['coadd_spread_model'] = self.cat['spread_model'][slist]
        st['flux_max'] = self.cat['flux_max'][slist]

        for i in xrange(ns):
            iobj=slist[i]
            icut=icutlist[i]

            ares,res1,res2,widths,max_pixel=self._process_star(iobj,icut)

            if widths[0] > 0:
                st['fw25_arcsec'][i] = PIX_SCALE*widths[0]
                st['fw50_arcsec'][i] = PIX_SCALE*widths[1]
                st['fw75_arcsec'][i] = PIX_SCALE*widths[2]

            st['max_pixel'][i] = max_pixel
            st['amflags'][i] = ares['whyflag']
            st['flags1'][i] = res1['flags']
            st['flags2'][i] = res2['flags']

            if ares['whyflag'] == 0:
                st['ampars'][i,:] = [1.0,ares['wrow'],ares['wcol'],
                                     ares['Irr'],ares['Irc'],ares['Icc']]
            if res1['flags'] == 0:
                st['pars1'][i,:] = res1['pars']

            if res2['flags'] == 0:
                st['pars2'][i,:] = res2['pars']
                st['offset_arcsec'][i] = self._get_offset_arcsec(res2['pars'])
                gmix=gmix_image.GMix(st['pars2'][i,:])
                debug=False
                #if st['offset_arcsec'][i] > 0.8:
                #    debug=True
                em2_widths = gmix_image.util.measure_gmix_width(gmix,
                                                                WIDTHS,
                                                                expand=8, debug=debug)
                st['fw25_arcsec_em2'][i] = PIX_SCALE*em2_widths[0]
                st['fw50_arcsec_em2'][i] = PIX_SCALE*em2_widths[1]
                st['fw75_arcsec_em2'][i] = PIX_SCALE*em2_widths[2]
                print >>stderr,'\t\t',st['fw50_arcsec'][i],st['fw50_arcsec_em2'][i]

        return st


    def _process_star(self, iobj, icutout):
        from gmix_image import GMixEMBoot
        import admom

        defres=({'whyflag':ZERO_WEIGHT_PIXELS},
                {'flags':ZERO_WEIGHT_PIXELS},
                {'flags':ZERO_WEIGHT_PIXELS},
                numpy.zeros(len(WIDTHS)) - 9999,
                -9999.0)

        im,ivar,cen_guess=self._get_star_data(iobj,icutout)

        if im is None:
            return defres
        
        max_pixel = im.max()

        widths = measure_image_width(im, WIDTHS)

        cen1_guess=cen_guess
        sig1_guess=sqrt(2)

        ares = admom.admom(im,
                           cen_guess[0],
                           cen_guess[1],
                           guess=sig1_guess)



        gm1=GMixEMBoot(im, 1, cen1_guess, sigma_guess=sig1_guess)
        res1=gm1.get_result()
        
        if res1['flags'] == 0:
            #print cen_guess
            #print res1['pars']
            sig2_guess=sqrt( (res1['pars'][3] + res1['pars'][5])/2. )
            cen2_guess=[res1['pars'][1], res1['pars'][2] ]
        else:
            sig2_guess=sig1_guess
            cen2_guess=cen1_guess

        gm2=GMixEMBoot(im, 2, cen2_guess, sigma_guess=sig2_guess)

        res2=gm2.get_result()

        if False and res2['flags'] != 0:
            import images
            images.multiview(im)
            resp=raw_input('hit enter (q to quit): ')
            if resp.lower() == 'q':
                stop


        return ares,res1,res2,widths,max_pixel

    def _get_star_data(self, iobj, icutout):
        """
        We use a fixed size apertures for all stars but only process stars for
        which there are no neighbors in that aperture and for which there are
        not zero weights in that aperture.

        The smallest cutout we do is 32x32 so we will use a sub region
        of that to allow for pixel shifts, 25x25.  Then the region is
        rowcen-r1, colcen+r1 where r1 is (25-1)/2 = 12
        """

        defres=None,None,[]

        box_size=29
        rsub=(box_size-1)//2

        image0=self.meds.get_cutout(iobj,icutout)
        wt0=self.meds.get_cutout(iobj,icutout,type='weight')
        seg0=self.meds.get_cutout(iobj,icutout,type='seg')

        rowcen=self.meds['cutout_row'][iobj,icutout]
        colcen=self.meds['cutout_col'][iobj,icutout]

        sid=seg0[rowcen,colcen]

        # fix up compression issue

        """
        w=numpy.where(seg == sid)
        if w[0].size ==0:
            return defres
        rowmin=w[0].min()
        rowmax=w[0].max()
        colmin=w[1].min()
        colmax=w[1].max()
        """

        rowmin = int(rowcen) - rsub
        rowmax = int(rowcen) + rsub
        colmin = int(colcen) - rsub
        colmax = int(colcen) + rsub

        if ((rowmin < 0)
                or (rowmax > image0.shape[0])
                or (colmin < 0)
                or (colmax > image0.shape[1]) ):
            print >>stderr,'    hit bounds'
            return defres


        rowcen -= rowmin
        colcen -= colmin

        image = image0[rowmin:rowmax+1, colmin:colmax+1]
        wt    =    wt0[rowmin:rowmax+1, colmin:colmax+1]
        seg   =   seg0[rowmin:rowmax+1, colmin:colmax+1]

        if False:
            import images
            import biggles
            tab=biggles.Table(1,2)
            w0=numpy.where(wt0 < 0.2*wt0.max())
            if w0[0].size != 0:
                image0[w0] = 0
            w=numpy.where(wt < 0.2*wt.max())
            if w[0].size != 0:
                image[w] = 0

            nl=0.05
            tab[0,0]=images.view(image0,nonlinear=nl,title='raw',show=False)
            tab[0,1]=images.view(image,nonlinear=nl,title='cut',show=False)
            tab.show()
            resp=raw_input('hit enter (q to quit): ')
            if resp.lower() == 'q':
                stop

        # we don't work with images that have zero weight
        # in the image anywere because the EM code doesn't
        # have that feature
        # have to deal with noisy "zero" weights
        # also no pixels from other objects in our box
        wt_logic  =  ( wt < 0.2*wt.max()  )
        seg_logic = ( (seg != sid) & (seg != 0) )
        w_wt  = numpy.where(wt_logic)
        w_seg = numpy.where(seg_logic)

        if w_wt[0].size > 0:
            print >>stderr,'    ',iobj,'has',w_wt[0].size,'zero weight pixels'
            return None,None,[]
        if w_seg[0].size > 0:
            print >>stderr,'    ',iobj,'has',w_seg[0].size,'neighbor pixels'
            return None,None,[]


        ivar=numpy.median(wt)
        return image, ivar, [rowcen,colcen]


    def _get_star_data_old(self, iobj, icutout):
        image=self.meds.get_cutout(iobj,icutout)
        wt=self.meds.get_cutout(iobj,icutout,type='weight')
        #seg=self.meds.get_cutout(iobj,icutout,type='seg')

        # we don't work with images that have zero weight
        # in the image anywere because the EM code doesn't
        # have that feature
        # have to deal with noisy "zero" weights
        w=numpy.where(wt < 0.2*wt.max())
        if w[0].size > 0:
            print >>stderr,'    ',iobj,'has',w[0].size,'zero weight pixels'
            return None,None,[]

        rowcen=self.meds['cutout_row'][iobj,icutout]
        colcen=self.meds['cutout_col'][iobj,icutout]

        ivar=numpy.median(wt)
        """
        sid=seg[rowcen,colcen]
        w=numpy.where(seg != sid)
        if w[0].size > 0:
            wt[w] = 0.0
        """
        return image, ivar, [rowcen,colcen]

    def _get_offset_arcsec(self, pars):
        igauss=0
        row1=pars[igauss*6 + 1]
        col1=pars[igauss*6 + 2]
        igauss=1
        row2=pars[igauss*6 + 1]
        col2=pars[igauss*6 + 2]

        offset=sqrt( (row1-row2)**2 + (col1-col2)**2 )
        offset_arcsec = offset*0.265

        return offset_arcsec

    def _find_stars_in_se(self, file_id):
        slist=[]
        icutlist=[]
        for i in self.wstar:
            if self.meds['ncutout'][i] > 1:
                w,=numpy.where(self.meds['file_id'][i,:] == file_id)
                if w.size > 0:
                    if w.size > 1:
                        raise ValueError("found file id more than once")

                    slist.append(i)
                    icutlist.append(w[0])

        return slist, icutlist

    def _select_stars(self, show=False):
        all_logic = (   (self.cat['flags']==0)
                      & (self.cat['spread_model'] > SPREAD_RANGE[0])
                      & (self.cat['spread_model'] < SPREAD_RANGE[1])
                      & (self.cat['mag_auto'] > MAG_RANGE[0])
                      & (self.cat['mag_auto'] < MAG_RANGE[1]) )
        star_logic = ( all_logic
                       & (self.cat['mag_auto'] > STAR_MAG_RANGE[0])
                       & (self.cat['mag_auto'] < STAR_MAG_RANGE[1])
                      & (self.cat['spread_model'] > STAR_SPREAD_RANGE[0])
                      & (self.cat['spread_model'] < STAR_SPREAD_RANGE[1]) )

        w,=numpy.where(all_logic)
        wstar,=numpy.where(star_logic)


        self.w=w
        self.wstar=wstar

        print >>stderr,'found',wstar.size,'stars'

    def do_mag_spread_plot(self, show=False):
        import biggles

        plt=biggles.FramedArray(2,1)


        w=self.w
        wstar=self.wstar

        mins=self.cat['mag_auto'][w].min()
        maxs=self.cat['mag_auto'][w].max()

        zeroc = biggles.Curve([mins,maxs],[0,0])
        all_pts = biggles.Points(self.cat['mag_auto'][w],
                                 self.cat['spread_model'][w],
                                 type='dot')
        star_pts = biggles.Points(self.cat['mag_auto'][wstar],
                                  self.cat['spread_model'][wstar],
                                  type='dot', color='red')

        plt[0,0].add(zeroc,all_pts, star_pts)
        plt[1,0].add(zeroc,all_pts, star_pts)

        plt.aspect_ratio=1
        plt.title=self.tileband
        plt.xlabel=r'$mag_{auto}$'
        plt.ylabel=r'spread model'
        plt.xrange=[15,25]
        plt.uniform_limits=0

        plt[0,0].yrange=[-0.01,0.04]
        plt[1,0].yrange=[-0.002,0.002]

        if show:
            plt.show()

        self.mag_size_plt=plt

        epsname=get_sg_plot_path(self.version, self.tileband)
        write_eps_and_convert(plt,epsname)

    def _set_tileband(self):
        bname=os.path.basename( self.meds_file )
        self.tileband='-'.join( ( bname.split('-') )[0:0+3] )
        
    def _load_data(self):
        print >>stderr,'opening:',self.meds_file
        self.meds=meds.MEDS(self.meds_file)
        self.info=self.meds.get_image_info()
        self.meta=self.meds.get_meta()

        self.bnames=[os.path.basename(f) for f in self.info['image_path']]

        self.cat_file=self.meta['coaddcat_file'][0]
        print >>stderr,'reading:',self.cat_file
        self.cat = fitsio.read(self.cat_file,lower=True)
            

    def _make_dirs(self):
        maind=get_tile_dir(self.version)
        make_dirs(maind)

    def _get_struct(self,n):
        l=map(len, self.bnames[1:])
        maxs = reduce(max, l)

        npars1=1*6
        npars2=2*6

        dt=[('tileband','S%d' % len(self.tileband)),
            ('number','i4'),
            ('icutout','i4'),
            ('file_id','i4'),
            ('sename','S%d' % maxs),
            ('coadd_mag_auto','f8'),
            ('coadd_spread_model','f8'),
            ('max_pixel','f4'),
            ('flux_max','f4'), # from sectractor
            ('flags1','i4'),
            ('flags2','i4'),
            ('amflags','i4'),
            ('pars1','f8',npars1),
            ('pars2','f8',npars2),
            ('fw25_arcsec','f8'),
            ('fw50_arcsec','f8'),
            ('fw75_arcsec','f8'),
            ('fw25_arcsec_em2','f8'),
            ('fw50_arcsec_em2','f8'),
            ('fw75_arcsec_em2','f8'),
            ('ampars','f8',npars1),
            ('offset_arcsec','f8')]

        st=numpy.zeros(n, dtype=dt)

        st['pars1']=-9999
        st['pars2']=-9999
        st['fw25_arcsec'] = -9999
        st['fw50_arcsec'] = -9999
        st['fw75_arcsec'] = -9999
        st['fw25_arcsec_em2'] = -9999
        st['fw50_arcsec_em2'] = -9999
        st['fw75_arcsec_em2'] = -9999
        st['offset_arcsec'] = 9999
        return st


   

def measure_image_width(image, thresh_vals, smooth=0.1, nsub=10, type='erf'):
    """
    Measure width at the given threshold using the specified method.
    
    e.g. 0.5 would be the FWHM

    You can send smooth= for the erf method and nsub= for the interp
    method.

    parameters
    ----------
    image: 2-d darray
        The image to measure
    thresh_vals: scalar or sequence
        threshold is, e.g. 0.5 to get a Full Width at Half max
    smooth: float
        erf method only.

        The smoothing scale for the erf.  This should be between 0 and 1. If
        you have noisy data, you might set this to the noise value or greater,
        scaled by the max value in the images.  Otherwise just make sure it
        smooths enough to avoid pixelization effects.
    nsub: int
        interpolation method only

        sub-pixel samples in each dimension

    output
    ------
    widths: scalar or ndarray

    method
    ------
    erf method:
        sqrt(Area)/pi where Area is,

            nim=image.image.max()
            arg =  (nim-thresh)/smooth
            vals = 0.5*( 1 + erf(arg) )
            area = vals.sum()
            width = 2*sqrt(area/pi)

    interp method:
        Linear interpolation
    """

    if type=='erf':
        return measure_image_width_erf(image, thresh_vals, smooth=smooth)
    else:
        return measure_image_width_interp(image, thresh_vals, nsub=nsub)

def measure_image_width_erf(image, thresh_vals, smooth=0.1):
    """
    Measure width at the given threshold using an erf to smooth the contour.
    
    e.g. 0.5 would be the FWHM

    parameters
    ----------
    image: 2-d darray
        The image to measure
    thresh_vals: scalar or sequence
        threshold is, e.g. 0.5 to get a Full Width at Half max
    smooth: float
        The smoothing scale for the erf.  This should be between 0 and 1. If
        you have noisy data, you might set this to the noise value or greater,
        scaled by the max value in the images.  Otherwise just make sure it
        smooths enough to avoid pixelization effects.

    output
    ------
    widths: scalar or ndarray
        sqrt(Area)/pi where Area is,

            nim=image.image.max()
            arg =  (nim-thresh)/smooth
            vals = 0.5*( 1 + erf(arg) )
            area = vals.sum()
            width = 2*sqrt(area/pi)
    """
    from numpy import array, sqrt, zeros, pi, where
    from scipy.special import erf

    if isinstance(thresh_vals, (list,tuple,numpy.ndarray)):
        is_seq=True
    else:
        is_seq=False

    thresh_vals=array(thresh_vals,ndmin=1,dtype='f8')

    nim = image.copy()
    maxval=image.max()
    nim *= (1./maxval)

    widths=zeros(len(thresh_vals))
    for i,thresh in enumerate(thresh_vals):
        arg = (nim-thresh)/smooth

        vals = 0.5*( 1+erf(arg) )
        area = vals.sum()
        widths[i] = 2*sqrt(area/pi)

    if is_seq:
        return widths
    else:
        return widths[0]

def measure_image_width_interp(image, thresh_vals, nsub=20):
    """
    e.g. 0.5 would be the FWHM

    parameters
    ----------
    image: 2-d darray
        The image to measure
    thresh_vals: scalar or sequence
        threshold is, e.g. 0.5 to get a Full Width at Half max
    nsub: float

    output
    ------
    widths: scalar or ndarray
    """
    from numpy import array, sqrt, zeros, pi, where
    from scipy.special import erf

    if isinstance(thresh_vals, (list,tuple,numpy.ndarray)):
        is_seq=True
    else:
        is_seq=False

    thresh_vals=array(thresh_vals,ndmin=1,dtype='f8')

    nim0 = image.copy()
    maxval=image.max()
    nim0 *= (1./maxval)

    nim = _make_interpolated_image(nim0, nsub)

    widths=zeros(len(thresh_vals))

    for i,thresh in enumerate(thresh_vals):
        w=where(nim > thresh)

        area = w[0].size
        widths[i] = 2*sqrt(area/pi)/nsub

    if is_seq:
        return widths
    else:
        return widths[0]

def _make_interpolated_image(im, nsub, order=1):
    """
    Make a new image linearly interpolated 
    on a nsubxnsub grid
    """
    # mgrid is inclusive at end when step
    # is complex and indicates number of
    # points
    import scipy.ndimage
    zimage=scipy.ndimage.zoom(im, nsub, order=order)
    if True:
        import images
        images.multiview(im)
        images.multiview(zimage)

    return zimage

def test_measure_image_width(fwhm=20.0, smooth=0.1, nsub=20, type='erf'):
    import gmix_image
    print 'testing type:',type
    sigma=fwhm/2.3548

    dim=2*5*sigma
    if (dim % 2)==0:
        dim+=1
    dims=[dim]*2
    cen=[(dim-1)//2]*2

    gm=gmix_image.GMix([1.0, cen[0], cen[1], sigma**2, 0.0, sigma**2])

    im=gmix_image.gmix2image(gm, dims)
    
    if type=='erf':
        widths = measure_image_width(im, 0.5, smooth=smooth)
    else:
        widths = measure_image_width_interp(im, 0.5, nsub=nsub)
    print 'true:',fwhm
    print 'meas:',widths
    print 'meas/true-1:',widths/fwhm-1
 
def make_dirs(*args):
    for d in args:
        if not os.path.exists(d):
            print >>stderr,'making dir:',d
            try:
                os.makedirs(d)
            except:
                pass

def write_eps_and_convert(plt, epsname, dpi=90):
    import converter
    d=os.path.dirname(epsname)

    make_dirs(d)

    print 'writing:',epsname
    plt.write_eps(epsname)
    converter.convert(epsname, dpi=dpi)

def get_output_dir():
    desdata=os.environ['DESDATA']
    return os.path.join(desdata, 'users/esheldon/mag-dependent-psf')

def get_tile_dir(version):
    d=get_output_dir()
    return os.path.join(d, 'bytile-%s' % version)

def get_tile_path(version, tileband):
    d=get_tile_dir(version)
    return os.path.join(d, '%s_models_%s.fits' % (tileband,version))

def get_sg_plot_path(version, tileband, ext='eps'):
    d=get_tile_dir(version)
    return os.path.join(d, '%s_sg_%s.%s' % (tileband,version,ext))

def get_exp_dir(version):
    d=get_output_dir()
    return os.path.join(d, 'byexp-%s' % version)


def get_exp_size_mag_plot(version, expname, xfield, type='normal', ext='eps'):
    d=get_exp_dir(version)
    if 'mag' in xfield:
        xstr='mag'
    elif 'flux_max' in xfield:
        xstr='maxflux'
    else:
        raise ValueError("bad xfield '%s'" % xfield)
    return os.path.join(d, '%s_size_%s_%s_%s.%s' % (expname,xstr,version,type,ext))
