from __future__ import print_function
import os
import numpy
import fitsio
import json

from . import files
from .files import DEFAULT_NPER

class ConcatError(Exception):
    """
    EM algorithm hit max iter
    """
    def __init__(self, value):
         self.value = value
    def __str__(self):
        return repr(self.value)


# need to fix up the images instead of this
from .constants import PIXSCALE2

class Concat(object):
    """
    Concatenate the split files

    This is the more generic interface
    """
    def __init__(self,
                 run,
                 config_file,
                 meds_files,
                 bands=None, # band names for each meds file
                 root_dir=None,
                 nper=DEFAULT_NPER,
                 sub_dir=None,
                 blind=True,
                 clobber=False):

        from . import files

        self.run=run
        self.config_file=config_file
        self.meds_file_list=meds_files
        self.nbands=len(meds_files)
        if bands is None:
            bands = [str(i) for i in xrange(self.nbands)]
        else:
            emess=("wrong number of bands: %d "
                   "instead of %d" % (len(bands),self.nbands))
            assert len(bands)==self.nbands,emess
        self.bands=bands

        self.sub_dir=sub_dir
        self.nper=nper
        self.blind=blind
        self.clobber=clobber

        self.config = files.read_yaml(config_file)

        self._files=files.Files(run, root_dir=root_dir)

        self.make_collated_dir()
        self.set_collated_file()

        self.load_meds()
        self.set_chunks()

        if self.blind:
            self.blind_factor = get_blind_factor()


    def verify(self):
        """
        just run through and read the data, verifying we can read it
        """

        for i,split in enumerate(self.chunk_list):

            print('\t%d/%d ' %(i+1,nchunk), end='')
            try:
                data, epoch_data, meta = self.read_chunk(split)
            except ConcatError as err:
                print("error found: %s" % str(err))

    def concat(self):
        """
        actually concatenate the data, and add any new fields
        """
        print('writing:',self.collated_file)

        if os.path.exists(self.collated_file) and not self.clobber:
            print('file already exists, skipping')
            return

        dlist=[]
        elist=[]

        nchunk=len(self.chunk_list)
        for i,split in enumerate(self.chunk_list):

            print('\t%d/%d ' %(i+1,nchunk), end='')
            data, epoch_data, meta = self.read_chunk(split)

            if self.blind:
                self.blind_data(data)

            dlist.append(data)
            if epoch_data.dtype.names is not None:
                elist.append(epoch_data)

        # note using meta from last file
        self._write_data(dlist, elist, meta)

    def _write_data(self, dlist, elist, meta):
        """
        write the data, first to a local file then staging out
        the the final location
        """
        import esutil as eu
        from .files import StagedOutFile

        data=eu.numpy_util.combine_arrlist(dlist)
        if len(elist) > 0:
            do_epochs=True
            epoch_data=eu.numpy_util.combine_arrlist(elist)
        else:
            do_epochs=False

        with StagedOutFile(self.collated_file, tmpdir=self.tmpdir) as sf:
            with fitsio.FITS(sf.path,'rw',clobber=True) as fits:
                fits.write(data,extname="model_fits")

                if do_epochs:
                    fits.write(epoch_data,extname='epoch_data')

                fits.write(meta,extname="meta_data")

        print('output is in:',self.collated_file)

    def blind_data(self,data):
        """
        multiply all shear type values by the blinding factor

        This also includes the Q values from B&A
        """
        models=self.config['fit_models']

        names=data.dtype.names
        for model in models:

            g_name='%s_g' % model
            Q_name='%s_Q' % model
            flag_name='%s_flags' % model

            w,=numpy.where(data[flag_name] == 0)
            if w.size > 0:
                if g_name in names:
                    data[g_name][w,:] *= self.blind_factor

                if Q_name in names:
                    data[Q_name][w,:] *= self.blind_factor

    def pick_epoch_fields(self, epoch_data0):
        """
        pick out some fields, add some fields, rename some fields
        """
        import esutil as eu

        wkeep,=numpy.where(epoch_data0['cutout_index'] >= 0)
        if wkeep.size==0:
            print("None found with cutout_index >= 0")
            print(epoch_data0['cutout_index'])
            return numpy.zeros(1)

        epoch_data0=epoch_data0[wkeep]

        dt=epoch_data0.dtype.descr

        names=epoch_data0.dtype.names
        ind=names.index('band_num')
        dt.insert( ind, ('band','S1') )

        epoch_data = numpy.zeros(epoch_data0.size, dtype=dt)
        eu.numpy_util.copy_fields(epoch_data0, epoch_data)

        for band_num in xrange(self.nbands):

            w,=numpy.where(epoch_data['band_num'] == band_num)
            if w.size > 0:
                epoch_data['band'][w] = self.bands[band_num]

        return epoch_data

    def pick_fields(self, data0, meta):
        """
        pick out some fields, add some fields, rename some fields
        """
        import esutil as eu

        nbands=self.nbands
        
        names=list( data0.dtype.names )
        dt=[tdt for tdt in data0.dtype.descr]

        flux_ind = names.index('psf_flux_err')
        dt.insert(flux_ind+1, ('psf_flux_s2n','f8',nbands) )
        names.insert(flux_ind+1,'psf_flux_s2n')

        dt.insert(flux_ind+2, ('psf_mag','f8',nbands) )
        names.insert(flux_ind+2,'psf_mag')

        models=self.config['fit_models']
        models = ['coadd_%s' % mod for mod in models] + models

        do_T=False
        for ft in models:

            s2n_name='%s_flux_s2n' % ft
            flux_ind = names.index('%s_flux' % ft)
            dt.insert(flux_ind+1, (s2n_name, 'f8', nbands) )
            names.insert(flux_ind+1,s2n_name)

            mag_name='%s_mag' % ft
            magf = (mag_name, 'f8', nbands)
            dt.insert(flux_ind+2, magf)
            names.insert(flux_ind+2, mag_name)

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
        eu.numpy_util.copy_fields(data0, data)

        all_models=['psf'] + models 
        for ft in all_models:
            if self.nbands==1:
                self.calc_mag_and_flux_stuff_scalar(data, meta, ft)
            else:
                for band in xrange(nbands):
                    self.calc_mag_and_flux_stuff(data, meta, ft, band)
        
        if do_T:
            self.add_T_info(data, models)
        return data

    def add_T_info(self, data, models):
        """
        Add T S/N etc.
        """
        for ft in models:
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




    def calc_mag_and_flux_stuff(self, data, meta, model, band):
        """
        Get magnitudes
        """

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

    def calc_mag_and_flux_stuff_scalar(self, data, meta, model):
        """
        Get magnitudes
        """

        flux_name='%s_flux' % model
        cov_name='%s_flux_cov' % model
        s2n_name='%s_flux_s2n' % model
        flag_name = '%s_flags' % model
        mag_name='%s_mag' % model

        data[mag_name][:] = -9999.
        data[s2n_name][:] = 0.0

        if model=='psf':
            w,=numpy.where(data[flag_name][:] == 0)
        else:
            w,=numpy.where(data[flag_name] == 0)

        if w.size > 0:
            flux = ( data[flux_name][w]/PIXSCALE2 ).clip(min=0.001)
            magzero=meta['magzp_ref'][0]
            data[mag_name][w] = magzero - 2.5*numpy.log10( flux )

            if model=='psf':
                flux=data['psf_flux'][w]
                flux_err=data['psf_flux_err'][w]
                w2,=numpy.where(flux_err > 0)
                if w2.size > 0:
                    flux=flux[w2]
                    flux_err=flux_err[w2]
                    data[s2n_name][w[w2]] = flux/flux_err
            else:
                flux=data[cov_name][w]
                flux_var=data[cov_name][w]

                w2,=numpy.where(flux_var > 0)
                if w.size > 0:
                    flux=flux[w2]
                    flux_err=numpy.sqrt(flux_var[w2])
                    data[s2n_name][w[w2]] = flux/flux_err



    def read_data(self, fname, split):
        """
        Read the chunk data
        """

        if not os.path.exists(fname):
            raise ConcatError("file not found: %s" % fname)

        print(fname)
        try:
            with fitsio.FITS(fname) as fobj:
                data0       = fobj['model_fits'][:]
                epoch_data0 = fobj['epoch_data'][:]
                meta        = fobj['meta_data'][:]
        except IOError as err:
            raise ConcatError(str(err))

        # watching for an old byte order bug
        expected_index = numpy.arange(split[0],split[1]+1)+1
        w,=numpy.where(data0['number'] != expected_index)
        if w.size > 0:
            raise ConcatError("number field is corrupted in file: %s" % fname)

        data = self.pick_fields(data0,meta)

        if epoch_data0.dtype.names is not None:
            epoch_data = self.pick_epoch_fields(epoch_data0)
        else:
            epoch_data = epoch_data0

            
        return data, epoch_data, meta

    def set_chunks(self):
        """
        set the chunks in which the meds file was processed
        """
        self.chunk_list=files.get_chunks(self.nrows, self.nper)

    def load_meds(self):
        """
        Load the associated meds files
        """
        import meds
        print('loading meds')

        self.meds_list=[]
        for fname in self.meds_file_list:
            m=meds.MEDS(fname)
            self.meds_list.append(m)

        self.nrows=self.meds_list[0]['id'].size

    def make_collated_dir(self):
        collated_dir = self._files.get_collated_dir()
        files.try_makedir(collated_dir)

    def set_collated_file(self):
        """
        set the output file and the temporary directory
        """
        if self.blind:
            extra='blind'
        else:
            extra=None
            
        self.collated_file = self._files.get_collated_file(sub_dir=self.sub_dir,
                                                           extra=extra)
        self.tmpdir=files.get_temp_dir()


    def read_chunk(self, split):
        """
        read data and epoch data from a given split
        """
        chunk_file=self._files.get_output_file(split,sub_dir=self.sub_dir)

        # continuing line from above
        print(chunk_file)
        data, epoch_data, meta=self.read_data(chunk_file, split)
        return data, epoch_data, meta


def get_tile_key(tilename,band):
    key='%s-%s' % (tilename,band)
    return key

def key_by_tile_band(data0, ftype):
    """
    Group files from a goodlist by tilename-band and sort the lists
    """
    print('grouping by tile/band')
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

    print('found %d tile/band combinations' % len(data))

    return data

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


