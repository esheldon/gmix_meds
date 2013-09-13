import os
import fitsio
import json

class TileConcat(object):
    """
    Concatenate the split files for a single tile
    """
    def __init__(self, run, tilename, ftype):
        import desdb
        import deswl

        self.run=run
        self.tilename=tilename
        self.ftype=ftype


        self.rc=deswl.files.Runconfig(run)
        self.nper=self.rc['nper']

        self.df=desdb.files.DESFiles()

        self.set_output_file()
        self.find_coadd_run()
        self.set_nrows()

    def find_coadd_run(self):
        """
        Get the coadd run from tilename and medsconf
        """
        import glob
        meds_dir = self.df.dir('meds_run',medsconf=self.rc['medsconf'])
        pattern = os.path.join(meds_dir,'*%s*' % self.tilename)

        print pattern
        flist=glob.glob(pattern)
        print flist
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
        import desdb

        out_dir = self.df.dir('wlpipe_collated', run=self.run)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        self.out_file = self.df.url('wlpipe_me_collated',
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
                data = fitsio.read(fname, ext="model_fits")

                if i==0:
                    meta=fitsio.read(fname, ext="meta_data")
                    fobj.write(data,extname="model_fits")
                else:
                    fobj["model_fits"].append(data)

            fobj.write(meta,extname="meta_data")

        print 'output is in:',out_file


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

def concat_tile(goodlist_file, tilename, band, ftype):
    """
    concatenate the files from a single band

    Only works when you have a goodlist obviously!  If
    you just want to concatenate a know file list, use
    the standalone gmix-meds-concat

    parameters
    ----------
    goodlist_file:
        The file holding the "good" processing data.  The
        file list will be extracted.
    tilename:
        coadd tile name
    band:
        filter
    ftype:
        e.g. 'lmfit'
    """
    with open(goodlist_file) as fobj:
        data0=json.load(fobj)

    data=key_by_tile_band(data0, ftype)

    key=get_tile_key(tilename, band)
    flist=data[key]

    tc=TileConcat(flist)
    tc.concat()

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


