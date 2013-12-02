import os
import numpy
import fitsio
import json
from . import lmfit

# need to fix up the images instead of this
PIXSCALE=0.265
PIXSCALE2=PIXSCALE**2

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
        self.config = load_config(self.rc['config'])

        self.nper=self.rc['nper']

        self.df=desdb.files.DESFiles()

        self.set_output_file()
        self.find_coadd_run()
        self.set_nrows()

        if self.blind:
            self.blind_factor = get_blind_factor()


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
        from deswl.generic import get_chunks, extract_start_end
        out_file=self.out_file
        print 'writing:',out_file

        nper=self.rc['nper']
        startlist,endlist=get_chunks(self.nrows,nper)
        nchunk=len(startlist)

        if os.path.exists(out_file) and not self.clobber:
            print 'file already exists, skipping'
            return

        with fitsio.FITS(out_file,'rw',clobber=True) as fobj:

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
                data = self.read_data(fname)

                if self.blind:
                    self.blind_data(data)

                if i==0:
                    meta=fitsio.read(fname, ext="meta_data")
                    fobj.write(data,extname="model_fits")
                else:
                    fobj["model_fits"].append(data)

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

    def pick_fields(self, data0, meta):
        import esutil
        nband = data0['psf_flux'].shape[1]

        dt=[]
        names=[]
        ftypes=[]
        for d in data0.dtype.descr:
            n=d[0]
            if ('psf1' not in n 
                    and n != 'processed'
                    and 'loglike' not in n
                    and 'aic' not in n
                    and 'bic' not in n
                    and 'dof' not in n
                    and 'fit_prob' not in n):
                dt.append(d)
                names.append(n)

        
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

        all_models=['psf'] + simple_models 
        for ft in all_models:
            for band in xrange(nband):
                self.calc_mag_and_flux_stuff(data, meta, ft, band)
        
        if do_T:
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


        return data

    def calc_mag_and_flux_stuff(self, data, meta, model, band):
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
        with fitsio.FITS(fname) as fobj:
            data0 = fobj["model_fits"][:]
            meta  = fobj["meta_data"][:]

        data = self.pick_fields(data0,meta)
            
        return data


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
