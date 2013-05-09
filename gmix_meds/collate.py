import os
import fitsio
import json

class TileConcat(object):
    """
    Concatenate the split files for a single tile
    """
    def __init__(self, flist):
        self.flist=flist
        self._set_out_file()

    def concat(self):
        out_file=self.out_file
        print 'writing:',out_file

        first=self.flist[0]
        meta=fitsio.read(first, ext="meta_data")

        with fitsio.FITS(out_file,'rw',clobber=True) as fobj:
            nf=len(self.flist)
            for i,fname in enumerate(self.flist):
                print '\t%d/%d %s' %(i+1,nf,fname)
                data = fitsio.read(fname, ext="model_fits")

                if i==0:
                    fobj.write(data,extname="model_fits")
                else:
                    fobj["model_fits"].append(data)

            fobj.write(meta,extname="meta_data")


    def _set_out_file(self):
        """
        fname is one of the split file names
        """

        fname=self.flist[0]

        dname=os.path.dirname(fname)
        bname=os.path.basename(fname)

        fs=bname.split('-')
        if len(fs) != 7:
            raise ValueError("name in wrong format: '%s'" % bname)

        out_file = '-'.join( fs[0:4] + fs[6:] )
        out_file = os.path.join(dname,out_file)
        
        self.out_file=out_file

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


