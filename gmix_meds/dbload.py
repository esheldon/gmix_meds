"""
todo

    - pick columns to keep
    - pick columns to index

    - ask todd to do his magic
"""
from __future__ import print_function
import os
import numpy

from . import files

def make_all_oracle_input(run, table_name, blind=True):
    """
    Make inputs for all tiles used in the specified run
    """

    tilenames = get_tilenames(run)
    ntiles=len(tilenames)
    print('found',ntiles,'tiles')
    for i,tilename in enumerate(tilenames):
        print("-"*70)
        print('%03d/%03d %s' % (i+1,ntiles,tilename))

        create=(i==0)
        make_oracle_input(run, tilename, table_name,
                          blind=blind, create=create)

def make_oracle_input_split(run, table_name, nsplit, split, blind=True):
    """
    Make inputs for all tiles used in the specified run
    """

    if split >= nsplit:
        raise ValueError("split should be < nsplit")

    tilenames = get_tilenames(run)
    ntiles_total=len(tilenames)
    print('found',ntiles_total,'tiles')

    ntiles_per=ntiles_total/nsplit
    nleft = ntiles_total % nsplit

    beg=split*ntiles_per
    if split < (nsplit-1):
        end=(split+1)*ntiles_per
    else:
        end=(split+1)*ntiles_per + nleft

    for i in xrange(beg,end):
        if tilename in _TILE_BLACKLIST:
            continue
        tilename=tilenames[i]
        print('%03d:%03d %s' % (i,end-1,tilename))

        create=(i==0)
        make_oracle_input(run, tilename, table_name,
                          blind=blind, create=create)

def make_oracle_input(run, tilename, table_name,
                      blind=True, create=False):
    """
    convenience function to create the oracle input
    """
    oim=OracleInputMaker(run, tilename, table_name,
                         blind=blind, create=create)

    oim.make_tile_input()

class OracleInputMaker(object):
    """
    Create the input csv and control file.  Potentially make the sql statement
    for table creation.
    """
    def __init__(self, run, tilename, table_name, ftype='mcmc',
                 blind=True, create=False):

        self._files=files.Files(run)

        self.run=run
        self.tilename=tilename
        self.ftype=ftype

        self.table_name=table_name

        self.blind=blind
        self.create=create

        self.set_info()

        self.epoch_name_map={'id':'coadd_objects_id',
                             'number':'coadd_object_number', # to match the database
                             'psf_fit_g':'psf_fit_e'}

        self.name_map={'id':     'coadd_objects_id',
                       'number': 'coadd_object_number',

                       'coadd_psfrec_g':'coadd_psfrec_e',

                       'coadd_exp_g':      'coadd_exp_e',
                       'coadd_exp_g_cov':  'coadd_exp_e_cov',
                       'coadd_exp_g_sens': 'coadd_exp_e_sens',
                       'coadd_dev_g':      'coadd_dev_e',
                       'coadd_dev_g_cov':  'coadd_dev_e_cov',
                       'coadd_dev_g_sens': 'coadd_dev_e_sens',

                       'psfrec_g':   'psfrec_e',

                       'exp_g':      'exp_e',
                       'exp_g_cov':  'exp_e_cov',
                       'exp_g_sens': 'exp_e_sens',
                       'dev_g':      'dev_e',
                       'dev_g_cov':  'dev_e_cov',
                       'dev_g_sens': 'dev_e_sens'}

    def make_tile_input(self):
        """
        Create the control file and potentially the create table file
        """
        import desdb
        import fitsio
        import time

        print("reading data:",self.fname)
        with fitsio.FITS(self.fname) as fobj:
            tmodel_fits=fobj['model_fits'][0]
            cols2keep=self.extract_keep_cols(tmodel_fits)

            print("    reading model fits")
            model_fits=fobj['model_fits'].read(columns=cols2keep)
            print("    reading epoch data")
            epoch_data=fobj['epoch_data'][:]

        model_fits=self.add_tilename(model_fits)

        rename_columns(model_fits, self.name_map)
        rename_columns(epoch_data, self.epoch_name_map)

        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        print()
        print("array2table model fits")
        desdb.array2table(model_fits,
                          self.table_name,
                          self.control_file,
                          bands=self.bands,
                          band_cols=self.band_cols,
                          create=self.create,
                          primary_key=self.primary_key)


        print()
        print("array2table epoch data")
        desdb.array2table(epoch_data,
                          self.epochs_table_name,
                          self.epochs_control_file,
                          create=self.create)

    def add_tilename(self, data):
        import esutil as eu

        print("adding tilename")

        names=list( data.dtype.names )
        index=names.index('number')

        dt=data.dtype.descr
        dt.insert(index+1, ('tilename','S12'))

        newdata=numpy.zeros(data.size, dtype=dt)

        eu.numpy_util.copy_fields(data, newdata)
        
        newdata['tilename'] = self.tilename

        return newdata

    def extract_keep_cols(self, data):
        """
        throw out columns that match certain patterns
        """

        drop_patterns=['coadd_gauss','logpars','tau','pars','pars_cov',
                       'psf1','processed','aic','bic']

        names2keep=[]
        for name in data.dtype.names:
            keep=True
            for pattern in drop_patterns:
                if pattern in name:
                    keep=False
                    break

            if keep:
                names2keep.append(name)

        return names2keep
        
    def set_info(self):
        """
        Set some info needed to do our work
        """
        import desdb
        import deswl

        self.epochs_table_name = get_epochs_table_name(self.table_name)

        if self.blind:
            out_ftype='wlpipe_me_collated_blinded'
        else:
            out_ftype='wlpipe_me_collated'

        self.rc=deswl.files.read_runconfig(self.run)


        if self.blind:
            extra='blind'
        else:
            extra=None

        self.fname = self._files.get_collated_file(sub_dir=self.tilename, extra=extra)

        '''
        self.df=desdb.files.DESFiles()
        self.fname = self.df.url(out_ftype,
                                 run=self.run,
                                 tilename=self.tilename,
                                 filetype=self.ftype,
                                 ext='fits')
        '''

        dirname=os.path.dirname(self.fname)
        bname=os.path.basename(self.fname)

        self.outdir=os.path.join( dirname, 'oracle' )
        #self.tmpdir=get_temp_dir()

        control_file=os.path.basename(bname).replace('.fits','.ctl')
        self.control_file=os.path.join(self.outdir, control_file)
        #self.temp_control_file=os.path.join(self.tmpdir, control_file)

        self.epochs_control_file=self.control_file.replace('.ctl','-epochs.ctl')
        #self.temp_epochs_control_file=self.temp_control_file.replace('.ctl','-epochs.ctl')
        #self.temp_epochs_pattern=\
        #    os.path.join(self.tmpdir, self.temp_epochs_control_file.replace('.ctl','*'))

        #self.temp_pattern=os.path.join(self.tmpdir, self.temp_control_file.replace('.ctl','*'))

        #print(self.temp_pattern)

        self.bands=self.rc['bands']
        self.band_cols=get_band_cols()

        self.primary_key='coadd_objects_id'

def add_indexes(table_name):
    """
    Add indexes to the appropriate columns
    """
    import desdb
    epochs_table_name = get_epochs_table_name(table_name)
    epochs_table_name_short = get_epochs_table_name(table_name,short=True)

    index_cols=get_index_cols()
    epoch_index_cols=get_epoch_index_cols()


    qt="create index {index_name} on {table_name}({col})"

    conn=desdb.Connection()
    curs = conn.cursor()

    for col in index_cols:
        index_name='{table_name}{col}idx'.format(table_name=table_name,
                                                   col=col)
        index_name=index_name.replace('_','')


        query = qt.format(index_name=index_name,
                          table_name=table_name,
                          col=col)
        print(query)

        curs.execute(query)

    for col in epoch_index_cols:
        index_name='{table_name}{col}idx'.format(table_name=epochs_table_name_short,
                                                 col=col)
        index_name=index_name.replace('_','')


        query = qt.format(index_name=index_name,
                          table_name=epochs_table_name,
                          col=col)
        print(query)

        curs.execute(query)

    curs.close()
    conn.close()

def get_epochs_table_name(table_name, short=False):
    """
    For a table get the corresponding epochs
    table name
    """
    if short:
        return '%se' % table_name
    else:
        return '%s_epochs' % table_name

def get_band_cols():
    colnames=['nimage_tot',
              'nimage_use',

              'coadd_psfrec_counts_mean',
              'coadd_psf_flags',
              'coadd_psf_flux',
              'coadd_psf_flux_err',
              'coadd_psf_chi2per',
              'coadd_psf_dof',

              'psfrec_counts_mean',
              'psf_flags',
              'psf_flux',
              'psf_flux_err',
              'psf_flux_s2n',
              'psf_mag',
              'psf_chi2per',
              'psf_dof',

              'coadd_exp_flux',
              'coadd_exp_flux_s2n',
              'coadd_exp_mag',
              'coadd_exp_flux_cov',

              'coadd_dev_flux',
              'coadd_dev_flux_s2n',
              'coadd_dev_mag',
              'coadd_dev_flux_cov',
 
              'exp_flux',
              'exp_flux_s2n',
              'exp_mag',
              'exp_flux_cov',

              'dev_flux',
              'dev_flux_s2n',
              'dev_mag',
              'dev_flux_cov']

    return colnames

def get_index_cols():
    return [
            'tilename',
            'coadd_object_number',
            'flags',

            'coadd_psf_flags_g',
            'coadd_psf_flags_r',
            'coadd_psf_flags_i',
            'coadd_psf_flags_z',

            # forgot to make psf mag for coadd
            #'coadd_psf_mag_g',
            #'coadd_psf_mag_r',
            #'coadd_psf_mag_i',
            #'coadd_psf_mag_z',

            'psf_flags_g',
            'psf_flags_r',
            'psf_flags_i',
            'psf_flags_z',
            #'psf_mag_g',
            #'psf_mag_r',
            #'psf_mag_i',
            #'psf_mag_z',

            'coadd_exp_flags',
            #'coadd_exp_chi2per',
            #'coadd_exp_mag_g',
            #'coadd_exp_mag_r',
            #'coadd_exp_mag_i',
            #'coadd_exp_mag_z',
            #'coadd_exp_s2n_w',
            #'coadd_exp_T_s2n',
            #'coadd_exp_e_1',
            #'coadd_exp_e_2',
            #'coadd_exp_arate',

            'coadd_dev_flags',
            #'coadd_dev_chi2per',
            #'coadd_dev_mag_g',
            #'coadd_dev_mag_r',
            #'coadd_dev_mag_i',
            #'coadd_dev_mag_z',
            #'coadd_dev_s2n_w',
            #'coadd_dev_T_s2n',
            #'coadd_dev_e_1',
            #'coadd_dev_e_2',
            #'coadd_dev_arate',

            'exp_flags',
            #'exp_chi2per',
            #'exp_mag_g',
            #'exp_mag_r',
            #'exp_mag_i',
            #'exp_mag_z',
            #'exp_s2n_w',
            #'exp_T_s2n',
            #'exp_e_1',
            #'exp_e_2',
            #'exp_arate',

            'dev_flags',
            #'dev_chi2per',
            #'dev_mag_g',
            #'dev_mag_r',
            #'dev_mag_i',
            #'dev_mag_z',
            #'dev_s2n_w',
            #'dev_T_s2n',
            #'dev_e_1',
            #'dev_e_2',
            #'dev_arate',
            ]

def get_epoch_index_cols():
    return ['coadd_objects_id',
            'image_id',
            'cutout_index',
            'band',
            'band_num',
            'psf_fit_flags']

_TILE_BLACKLIST=['DES0503-6414', 'DES0557-6122']
def get_tilenames(run):
    """
    Get all associated tilenames
    """
    import desdb
    import deswl

    rc=deswl.files.read_runconfig(run)
    releases=rc['dataset']

    if releases=="testbed":
        runs=desdb.files.get_testbed_runs(rc)
    else:
        bands=rc['bands']
        runs=desdb.files.get_release_runs(releases, withbands=bands)

    tiles=[]
    for run in runs:
        rs=run.split('_')
        tilename=rs[1]
        if tilename not in _TILE_BLACKLIST:
            tiles.append(tilename)

    tiles.sort()
    return tiles


def rename_columns(arr, name_map):
    names=list( arr.dtype.names )
    for i,n in enumerate(names):
        if n in name_map:
            names[i] = name_map[n]

    arr.dtype.names=tuple(names)

