from sys import stderr
import os
import numpy
from numpy import sqrt
import fitsio
import meds
import gmix_image

ZERO_WEIGHT_PIXELS=2**30


MAG_RANGE = [12.,25.]
SPREAD_RANGE =[-0.01,0.04]
STAR_MAG_RANGE = [15.5,19.0]
STAR_SPREAD_RANGE = [-0.002,0.002]

def collate_results(out_file):
    import glob
    d=get_output_dir()

    flist=glob.glob(d+'/DES*/*.fits')

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
    def __init__(self, collated_file):
        print collated_file
        self.data=fitsio.read(collated_file)

        self.offset_max=0.25

        self._extract_by_exp()

    def doplots(self, show=False):
        nexp=len(self.expdict)
        i=1
        for expname,exp_objs in self.expdict.iteritems():
            print >>stderr,'%d/%d' % (i,nexp)

            self._process_exp(expname,exp_objs,show=show)
            i += 1

    def _process_exp(self, expname, exp_objs,show=False):
        import biggles
        import pcolors
        print >>stderr,'found',len(exp_objs),'ccds in',expname
       
        ccds = list(exp_objs.keys())
        nccd = len(ccds)

        plt=biggles.FramedArray(2,1)
        plt.title='%s  nccd: %d' % (expname,nccd)
        plt.yrange=[0.7,1.6]
        plt.xrange=[15.4,19.2]
        plt.uniform_limits=1

        plt1=biggles.FramedPlot()
        plt2=biggles.FramedPlot()
        xlabel=r'$mag_{auto}$'
        ylabel=r'$FWHM [arcsec]$'

        plt.xlabel=xlabel
        plt.ylabel=ylabel


        colors=pcolors.rainbow(nccd)

        for i,ccd in enumerate(ccds):
            wccd = numpy.array(exp_objs[ccd])
            color = colors[i]

            nw=len(wccd)
            #print '    %d in ccd %d' % (nw,ccd)

            w1, = numpy.where(self.data['flags1'][wccd]==0)
            if w1.size != 0:
                w1=wccd[w1]
                self._add_to_plot(plt[0,0], 1, w1, color)

            offsets=self.data['offset_arcsec'][wccd]
            w2, = numpy.where(  (self.data['flags2'][wccd]==0)
                              & (offsets < self.offset_max))
            if w2.size != 0:
                w2=wccd[w2]
                self._add_to_plot(plt[1,0], 2, w2, color)

        lab1=biggles.PlotLabel(0.1,0.9,'ngauss: 1',halign='left')
        lab2_1=biggles.PlotLabel(0.1,0.9, "ngauss: 2", halign='left')
        lab2_2=biggles.PlotLabel(0.1,0.8,
                               "offset < %.2f''" % self.offset_max,
                               halign='left')
        plt[0,0].add(lab1)
        plt[1,0].add(lab2_1,lab2_2)

        if show:
            plt.show()
            key=raw_input('hit a key (q to quit): ')
            if key.lower() == 'q':
                stop

        epsname=get_exp_size_mag_plot(expname)
        write_eps_and_convert(plt,epsname)


    def _add_to_plot(self, plt, ngauss, w, color):
        import biggles

        fwhm=numpy.zeros(w.size)
        for i,wi in enumerate(w):

            if ngauss==1:
                gmix=gmix_image.GMix(self.data['pars1'][wi,:])
            else:
                gmix=gmix_image.GMix(self.data['pars2'][wi,:])

            T=gmix.get_T()
            
            fwhm[i] = 0.265*2.3548*sqrt(T/2.)

        pts=biggles.Points(self.data['coadd_mag_auto'][w],
                           fwhm, type='dot',
                           color=color)
        plt.add(pts)

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
    def __init__(self, meds_file):

        self.meds_file=meds_file

        self._load_data()
        self._set_tileband()
        self._make_dirs()
        self._select_stars()

    def measure_stars(self):

        out_file=get_tile_path(self.tileband)
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
        ns=len(slist)
        st=self._get_struct(ns)
        st['tileband'] = self.tileband
        st['number'] = slist
        st['icutout'] = icutlist
        st['file_id'] = file_id
        st['sename']=self.bnames[file_id]
        st['coadd_mag_auto'] = self.cat['mag_auto'][slist]
        st['coadd_spread_model'] = self.cat['spread_model'][slist]

        for i in xrange(ns):
            iobj=slist[i]
            icut=icutlist[i]

            res1,res2=self._process_star(iobj,icut)

            st['flags1'][i] = res1['flags']
            st['flags2'][i] = res2['flags']

            if res1['flags'] == 0:
                st['pars1'][i,:] = res1['pars']

            if res2['flags'] == 0:
                st['pars2'][i,:] = res2['pars']
                st['offset_arcsec'][i] = self._get_offset_arcsec(res2['pars'])
        return st


    def _process_star(self, iobj, icutout):
        from gmix_image import GMixEMBoot

        im,ivar,cen_guess=self._get_star_data(iobj,icutout)

        if im is None:
            return [{'flags':ZERO_WEIGHT_PIXELS}]*2
        
        cen1_guess=cen_guess
        sig1_guess=sqrt(2)
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


        return res1,res2

    def _get_star_data(self, iobj, icutout):
        defres=None,None,[]

        image0=self.meds.get_cutout(iobj,icutout)
        wt0=self.meds.get_cutout(iobj,icutout,type='weight')
        seg=self.meds.get_cutout(iobj,icutout,type='seg')
        rowcen=self.meds['cutout_row'][iobj,icutout]
        colcen=self.meds['cutout_col'][iobj,icutout]

        sid=seg[rowcen,colcen]
        w=numpy.where(seg == sid)
        if w[0].size ==0:
            return defres

        rowmin=w[0].min()
        rowmax=w[0].max()
        colmin=w[1].min()
        colmax=w[1].max()

        rowcen -= rowmin
        colcen -= colmin

        image=image0[rowmin:rowmax+1, colmin:colmax+1]
        wt=wt0[rowmin:rowmax+1, colmin:colmax+1]

        if False:
            import images
            images.view(image0,title='raw')
            images.view(image,title='cut')
            resp=raw_input('hit enter (q to quit): ')
            if resp.lower() == 'q':
                stop

        # we don't work with images that have zero weight
        # in the image anywere because the EM code doesn't
        # have that feature
        # have to deal with noisy "zero" weights
        w=numpy.where(wt < 0.2*wt.max())
        if w[0].size > 0:
            print >>stderr,'    ',iobj,'has',w[0].size,'zero weight pixels'
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

        epsname=get_sg_plot_path(self.tileband)
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
        maind=get_tile_dir()
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
            ('flags1','i4'),
            ('flags2','i4'),
            ('pars1','f8',npars1),
            ('pars2','f8',npars2),
            ('offset_arcsec','f8')]

        st=numpy.zeros(n, dtype=dt)

        st['pars1']=-9999
        st['pars2']=-9999
        st['offset_arcsec'] = 9999
        return st

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

def get_tile_dir():
    d=get_output_dir()
    return os.path.join(d, 'bytile')

def get_tile_path(tileband):
    d=get_tile_dir()
    return os.path.join(d, '%s_models.fits' % tileband)

def get_sg_plot_path(tileband, ext='eps'):
    d=get_tile_dir()
    return os.path.join(d, '%s_sg.%s' % (tileband,ext))

def get_exp_dir():
    d=get_output_dir()
    return os.path.join(d, 'byexp')


def get_exp_size_mag_plot(expname, ext='eps'):
    d=get_exp_dir()
    return os.path.join(d, '%s_size_mag.%s' % (expname,ext))
