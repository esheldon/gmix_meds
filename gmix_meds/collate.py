import os
import numpy
import fitsio
import json
from . import lmfit

# need to fix up the images instead of this
from .constants import PIXSCALE, PIXSCALE2

def load_config(name):
    path = '$GMIX_MEDS_DIR/share/config/%s.yaml' % name
    path = os.path.expandvars(path)

    return load_config_path(path)

def load_config_path(fname):
    import yaml
    with open(fname) as fobj:
        data=yaml.load(fobj)
    return data

class TileConcat(object):
    """
    Concatenate the split files for a single tile
    """
    def __init__(self, run, tilename, ftype, blind=False, clobber=False):
        import desdb
        import deswl

        self.run=run
        self.tilename=tilename
        self.ftype=ftype
        self.blind=blind
        self.clobber=clobber

        self.rc=deswl.files.Runconfig(run)
        self.bands=self.rc['band']
        self.nbands=len(self.bands)
        self.config = load_config(self.rc['config'])

        self.nper=self.rc['nper']

        self.df=desdb.files.DESFiles()

        self.set_output_file()
        self.find_coadd_run()

        self.load_meds()
        self.find_image_ids()
        self.set_nrows()

        self.set_coadd_objects_info()

        if self.blind:
            self.blind_factor = get_blind_factor()

    def load_meds(self):
        """
        Load the associated meds files
        """
        import meds
        print 'loading meds'

        self.meds_list=[]
        for band in self.bands:
            fname=self.df.url('meds',
                              tilename=self.tilename,
                              band=band,
                              medsconf=self.rc['medsconf'],
                              coadd_run=self.coadd_run)
            m=meds.MEDS(fname)
            self.meds_list.append(m)

    def find_image_ids(self):
        """
        Get the image id from the database for each of the 
        SE exposures.  Note we keep a slot for the coadd but
        it is not used
        """
        import desdb

        print 'getting image ids'
        conn=desdb.Connection()

        for band in xrange(self.nbands):
            m=self.meds_list[band]

            nimage=m._image_info.size
            m._image_ids = numpy.zeros(nimage,dtype='i8')

            for file_id in xrange(1,nimage):
                fname=m._image_info['image_path'][file_id]
                ii=extract_red_info(fname)
                query="""
                select
                    id
                from
                    location
                where
                    run='{run}'
                    and exposurename='{expname}'
                    and filetype='red'
                    and ccd={ccd}"""
                query=query.format(run=ii['run'],
                                   expname=ii['expname'],
                                   ccd=ii['ccd'])
                res=conn.quick(query)
                if len(res) != 1:
                    raise ValueError("expected one image match, "
                                     "got %s" % repr(res))
                m._image_ids[file_id] = res[0]['id']

    def find_coadd_run(self):
        """
        Get the coadd run from tilename and medsconf
        """
        import glob
        meds_dir = self.df.dir('meds_run',medsconf=self.rc['medsconf'])
        pattern = os.path.join(meds_dir,'*%s*' % self.tilename)

        flist=glob.glob(pattern)
        if len(flist) != 1:
            raise RuntimeError("expected 1 dir, found %d" % len(flist))

        self.coadd_run = os.path.basename(flist[0])


    def set_nrows(self):
        nper=self.rc['nper']

        meds_filename=self.df.url('meds',
                                  tilename=self.tilename,
                                  band='i',
                                  medsconf=self.rc['medsconf'],
                                  coadd_run=self.coadd_run)
        with fitsio.FITS(meds_filename) as fobj:
            self.nrows=fobj['object_data'].get_nrows()

    def set_output_file(self):

        out_dir = self.df.dir('wlpipe_collated', run=self.run)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        if self.blind:
            out_ftype='wlpipe_me_collated_blinded'
        else:
            out_ftype='wlpipe_me_collated'
            
        self.out_file = self.df.url(out_ftype,
                                    run=self.run,
                                    tilename=self.tilename,
                                    filetype=self.ftype,
                                    ext='fits')

    def concat(self):
        import esutil as eu
        from deswl.generic import get_chunks, extract_start_end
        out_file=self.out_file
        print 'writing:',out_file

        nper=self.rc['nper']
        startlist,endlist=get_chunks(self.nrows,nper)
        nchunk=len(startlist)

        if os.path.exists(out_file) and not self.clobber:
            print 'file already exists, skipping'
            return

        dlist=[]
        elist=[]
        npsf=0

        for i in xrange(nchunk):

            start=startlist[i]
            end=endlist[i]
            sstr,estr = extract_start_end(start=start, end=end)

            fname=self.df.url('wlpipe_me_split',
                              run=self.run,
                              tilename=self.tilename,
                              start=sstr,
                              end=estr,
                              filetype=self.ftype,
                              ext='fits')

            print '\t%d/%d %s' %(i+1,nchunk,fname)
            data, epoch_data = self.read_data(fname)

            if self.blind:
                self.blind_data(data)

            dlist.append(data)
            if epoch_data.dtype.names is not None:
                elist.append(epoch_data)

        data=eu.numpy_util.combine_arrlist(dlist)
        if len(elist) > 0:
            do_epochs=True
            epoch_data=eu.numpy_util.combine_arrlist(elist)
        else:
            do_epochs=False

        with fitsio.FITS(out_file,'rw',clobber=True) as fobj:
            meta=fitsio.read(fname, ext="meta_data")
            fobj.write(data,extname="model_fits")

            if do_epochs > 0:
                fobj.write(epoch_data,extname='epoch_data')

            fobj.write(meta,extname="meta_data")

        print 'output is in:',out_file

    def blind_data(self,data):
        simple_models=self.config.get('simple_models',lmfit.SIMPLE_MODELS_DEFAULT )

        for fit_type in simple_models:
            g_name='%s_g' % fit_type
            flag_name='%s_flags' % fit_type

            if flag_name in data.dtype.names:
                w,=numpy.where(data[flag_name] == 0)
                if w.size > 0:
                    data[g_name][w,:] *= self.blind_factor

    def pick_epoch_fields(self, epoch_data0):
        """
        pick out some fields, add some fields, rename some fields
        """
        import esutil
        name_map={'number':'coadd_object_number', # to match the database
                  'psf_fit_g':'psf_fit_e'}
        rename_columns(epoch_data0, name_map)

        # we can loosen this when we store the cutout sub-id
        # in the output file....  right now file_id can be
        # not set
        wkeep,=numpy.where(epoch_data0['coadd_object_number'] > 0)
        if wkeep.size==0:
            return numpy.zeros(1)

        epoch_data0=epoch_data0[wkeep]

        dt=epoch_data0.dtype.descr

        names=epoch_data0.dtype.names
        ind=names.index('band_num')
        dt.insert( ind, ('band','S1') )
        dt =  [('coadd_objects_id','i8')] + dt

        ind=names.index('coadd_object_number')
        dt.insert( ind, ('image_id','i8') )

        epoch_data = numpy.zeros(epoch_data0.size, dtype=dt)
        esutil.numpy_util.copy_fields(epoch_data0, epoch_data)

        self.add_coadd_objects_id(epoch_data)

        for band_num in xrange(self.nbands):
            w,=numpy.where(epoch_data['band_num'] == band_num)
            if w.size > 0:

                epoch_data['band'][w] = self.bands[band_num]

                m=self.meds_list[band_num]
                file_ids=epoch_data['file_id'][w]
                epoch_data['image_id'][w] = m._image_ids[file_ids]

        return epoch_data

    def pick_fields(self, data0, meta):
        """
        pick out some fields, add some fields, rename some fields
        """
        import esutil
        nband = data0['psf_flux'].shape[1]

        name_map={'number':     'coadd_object_number',
                  'exp_g':      'exp_e',
                  'exp_g_cov':  'exp_e_cov',
                  'exp_g_sens': 'exp_shear_sens',
                  'dev_g':      'dev_e',
                  'dev_g_cov':  'dev_e_cov',
                  'dev_g_sens': 'dev_shear_sens'}
        rename_columns(data0, name_map)

        dt=[]
        names=[]
        ftypes=[]
        for d in data0.dtype.descr:
            n=d[0]
            if ('psf1' not in n 
                    and n != 'processed'
                    and 'aic' not in n
                    and 'bic' not in n):
                dt.append(d)
                names.append(n)
        
        dt = [('coadd_objects_id','i8'), ('tilename','S12')] + dt
        
        flux_ind = names.index('psf_flux')
        dt.insert(flux_ind+1, ('psf_flux_s2n','f8',nband) )
        dt.insert(flux_ind+2, ('psf_mag','f8',nband) )

        simple_models=self.config.get('simple_models',
                                      lmfit.SIMPLE_MODELS_DEFAULT )

        do_T=False
        if 'simple' in self.config['fit_types']:
            for ft in simple_models:
                flux_ind = names.index('%s_flux' % ft)
                dt.insert(flux_ind+1, ('%s_flux_s2n' % ft, 'f8', nband) )

                magf = ('%s_mag' % ft, 'f8', nband)
                dt.insert(flux_ind+2, magf)

                Tn = '%s_T' % ft
                Ten = '%s_err' % Tn
                Ts2n = '%s_s2n' % Tn

                if Tn not in data0.dtype.names:
                    fadd=[(Tn,'f8'),
                          (Ten,'f8'),
                          (Ts2n,'f8')]
                    ind = names.index('%s_flux_cov' % ft)
                    for f in fadd:
                        dt.insert(ind+1, f)
                        names.insert(ind+1, f[0])
                        ind += 1

                    do_T=True


        data=numpy.zeros(data0.size, dtype=dt)
        esutil.numpy_util.copy_fields(data0, data)
        data['tilename'] = self.tilename

        self.add_coadd_objects_id(data)

        all_models=['psf'] + simple_models 
        for ft in all_models:
            for band in xrange(nband):
                self.calc_mag_and_flux_stuff(data, meta, ft, band)
        
        if do_T:
            self.add_T_info(data, simple_models)
        return data

    def add_T_info(self, data, simple_models):
        """
        Add T S/N etc.
        """
        for ft in simple_models:
            pn = '%s_pars' % ft
            pcn = '%s_pars_cov' % ft

            Tn = '%s_T' % ft
            Ten = '%s_err' % Tn
            Ts2n = '%s_s2n' % Tn
            fn='%s_flags' % ft

            data[Tn][:]   = -9999.0
            data[Ten][:]  =  9999.0
            data[Ts2n][:] = -9999.0

            Tcov=data[pcn][:,4,4]
            w,=numpy.where( (data[fn] == 0) & (Tcov > 0.0) )
            if w.size > 0:
                data[Tn][w]   = data[pn][w, 4]
                data[Ten][w]  =  numpy.sqrt(Tcov[w])
                data[Ts2n][w] = data[Tn][w]/data[Ten][w]



    def add_coadd_objects_id(self, data):
        """
        match by coadd_object_number and add the coadd_objects_id
        """
        import esutil
        cdata=self.coadd_objects_data

        h,rev=esutil.stat.histogram(data['coadd_object_number'],
                                    min=self.min_object_number,
                                    max=self.max_object_number,
                                    rev=True)
        nmatch=0

        n=cdata.size
        for i in xrange(n):
            if rev[i] != rev[i+1]:
                w=rev[ rev[i]:rev[i+1] ]
                coadd_objects_id = cdata['coadd_objects_id'][i]
                data['coadd_objects_id'][w] = coadd_objects_id

                nmatch += w.size

        if nmatch != data.size:
            raise ValueError("only %d/%d matched by "
                             "coadd_object_number" % (nmatch,data.size))

    def calc_mag_and_flux_stuff(self, data, meta, model, band):
        """
        Get magnitudes
        """
        nband = data['psf_flux'].shape[1]

        flux_name='%s_flux' % model
        cov_name='%s_flux_cov' % model
        s2n_name='%s_flux_s2n' % model
        flag_name = '%s_flags' % model
        mag_name='%s_mag' % model

        data[mag_name][:,band] = -9999.
        data[s2n_name][:,band] = 0.0

        if model=='psf':
            w,=numpy.where(data[flag_name][:,band] == 0)
        else:
            w,=numpy.where(data[flag_name] == 0)

        if w.size > 0:
            for band in xrange(nband):
                flux = ( data[flux_name][w,band]/PIXSCALE2 ).clip(min=0.001)
                magzero=meta['magzp_ref'][band]
                data[mag_name][w,band] = magzero - 2.5*numpy.log10( flux )

                if model=='psf':
                    flux=data['psf_flux'][w,band]
                    flux_err=data['psf_flux_err'][w,band]
                    w2,=numpy.where(flux_err > 0)
                    if w2.size > 0:
                        flux=flux[w2]
                        flux_err=flux_err[w2]
                        data[s2n_name][w[w2],band] = flux/flux_err
                else:
                    flux=data[cov_name][w,band,band]
                    flux_var=data[cov_name][w,band,band]

                    w2,=numpy.where(flux_var > 0)
                    if w.size > 0:
                        flux=flux[w2]
                        flux_err=numpy.sqrt(flux_var[w2])
                        data[s2n_name][w[w2], band] = flux/flux_err


    def read_data(self, fname):
        """
        Read the chunk data
        """
        with fitsio.FITS(fname) as fobj:
            data0       = fobj['model_fits'][:]
            epoch_data0 = fobj['epoch_data'][:]
            meta        = fobj['meta_data'][:]

        coadd=fitsio.read(meta['coaddcat_file'][0],lower=True)
        data = self.pick_fields(data0,meta)

        if epoch_data0.dtype.names is not None:
            epoch_data = self.pick_epoch_fields(epoch_data0)
        else:
            epoch_data = epochs_data0

            
        return data, epoch_data


    def set_coadd_objects_info(self):
        import desdb
        print 'getting coadd_objects ids'

        query="""
        select
            co.coadd_objects_id, co.object_number as coadd_object_number
        from
            coadd_objects co, coadd c
        where
            co.imageid_g=c.id
            and c.band='g'
            and c.run='{run}'
        order by
            coadd_object_number
        """.format(run=self.coadd_run)

        conn=desdb.Connection()
        res=conn.quick(query, array=True)
        conn.close()

        nmax=res['coadd_object_number'].max()
        if nmax != res.size:
            raise ValueError("some missing object_number, got "
                             "max %d for nobj %d" % (nmax,res.size))


        self.coadd_objects_data=res
        self.min_object_number=res['coadd_object_number'].min()
        self.max_object_number=res['coadd_object_number'].max()

    def read_meds_meta(self, meds_files):
        """
        get the meds metadata
        """
        pass



def concat_all(goodlist_file, badlist_file, ftype):
    """
    Concatenate all tiles in the list

    parameters
    ----------
    goodlist:
        File holding the good processing
    badlist:
        File holding the bad processing. Contents
        must be empty
    ftype:
        File type to concatenate, e.g. 'lmfit'
    """
    data0 = _load_and_check_lists(goodlist_file, badlist_file)

    data=key_by_tile_band(data0, ftype)

    keys=list(data.keys())
    ntot=len(keys)
    for i,key in enumerate(keys):
        print '%d/%d' % (i+1,ntot)
        flist = data[key]
        tc=TileConcat(flist)
        tc.concat()




def get_tile_key(tilename,band):
    key='%s-%s' % (tilename,band)
    return key

def key_by_tile_band(data0, ftype):
    """
    Group files from a goodlist by tilename-band and sort the lists
    """
    print 'grouping by tile/band'
    data={}

    for d in data0:
        fname=d['output_files'][ftype]
        key=get_tile_key(d['tilename'],d['band'])

        if key not in data:
            data[key] = [fname]
        else:
            data[key].append(fname)

    for key in data:
        data[key].sort()

    print 'found %d tile/band combinations' % len(data)

    return data


def _load_and_check_lists(goodlist,badlist):
    with open(goodlist) as fobj:
        good_data=json.load(fobj)
    with open(badlist) as fobj:
        bad_data=json.load(fobj)
    ngood=len(good_data)
    nbad=len(bad_data)
    if len(bad_data) != 0:
        raise ValueError("there were some failures: "
                         "%d/%d" % (nbad,nbad+ngood))

    return good_data


def get_blind_factor():
    """
    by joe zuntz
    """
    import sys
    import hashlib

    code_phrase = "DES is blinded"

    #hex number derived from code phrase
    m = hashlib.md5(code_phrase).hexdigest()
    #convert to decimal
    s = int(m, 16)
    # last 8 digits
    f = s%100000000
    # turn 8 digit number into value between 0 and 1
    g = f*1e-8
    #get value between 0.9 and 1
    return 0.9 + 0.1*g

def rename_columns(arr, name_map):
    names=list( arr.dtype.names )
    for i,n in enumerate(names):
        if n in name_map:
            names[i] = name_map[n]

    arr.dtype.names=tuple(names)

def extract_red_info(path):
    parts=path.split('/')

    run = parts[-4]
    expname=parts[-2]

    bname=parts[-1]
    bname=bname[0: bname.index('.')]

    ccd=int(  ( bname.split('_') )[2]  )

    return {'run':run,
            'expname':expname,
            'ccd':ccd}
