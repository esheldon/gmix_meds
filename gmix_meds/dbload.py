import os

def make_all_oracle_input(run, ftype, table_name, blind=True):
    """
    Make inputs for all tiles used in the specified run
    """

    tilenames = get_tilenames(run)
    ntiles=len(tilenames)
    print 'found',ntiles,'tiles'
    for i,tilename in enumerate(tilenames):
        print '%03d/%03d %s' % (i+1,ntiles,tilename)

        create=(i==0)
        make_oracle_input(run, tilename, ftype, table_name,
                          blind=blind, create=create)

def make_oracle_input(run, tilename, ftype, table_name,
                      blind=True, create=False):
    """
    convenience function to create the oracle input
    """
    oim=OracleInputMaker(run, tilename, ftype, table_name,
                         blind=blind, create=create)

    oim.make_tile_input()

class OracleInputMaker(object):
    """
    Create the input csv and control file.  Potentially make the sql statement
    for table creation.
    """
    def __init__(self, run, tilename, ftype, table_name,
                 blind=True, create=False):
        self.run=run
        self.tilename=tilename
        self.ftype=ftype

        self.table_name=table_name

        self.blind=blind
        self.create=create

        self.set_info()


    def make_tile_input(self):
        """
        Create the control file and potentially the create table file
        """
        import desdb
        import fitsio

        with fitsio.FITS(self.fname) as fobj:
            model_fits=fobj['model_fits'][:]
            epoch_data=fobj['epoch_data'][:]

        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        desdb.array2table(model_fits,
                          self.table_name,
                          self.control_file,
                          bands=self.bands,
                          band_cols=self.band_cols,
                          create=self.create,
                          primary_key=self.primary_key)

        desdb.array2table(epoch_data,
                          self.epochs_table_name,
                          self.epochs_control_file,
                          create=self.create)

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

        self.rc=deswl.files.Runconfig(self.run)

        self.df=desdb.files.DESFiles()
        self.fname = self.df.url(out_ftype,
                                 run=self.run,
                                 tilename=self.tilename,
                                 filetype=self.ftype,
                                 ext='fits')

        dirname=os.path.dirname(self.fname)
        bname=os.path.basename(self.fname)

        self.outdir=os.path.join( dirname, 'oracle' )

        control_file=os.path.basename(bname).replace('.fits','.ctl')
        self.control_file=os.path.join(self.outdir, control_file)

        self.epochs_control_file=self.control_file.replace('.ctl','-epochs.ctl')


        self.bands=self.rc['band']
        self.band_cols=get_band_cols()

        self.primary_key='coadd_objects_id'

def add_indexes(table_name):
    """
    Add indexes to the appropriate columns
    """
    import desdb
    epochs_table_name = get_epochs_table_name(table_name)

    index_cols=get_index_cols()
    epoch_index_cols=get_epoch_index_cols()

    qt="create index {col}_idx on {table_name}({col})"

    conn=desdb.Connection()
    curs = conn.cursor()
    for col in index_cols:
        query = qt.format(table_name=table_name, col=col)
        print query

        curs.execute(query)

    for col in epoch_index_cols:
        query = qt.format(table_name=epochs_table_name, col=col)
        print query

        curs.execute(query)

    curs.close()
    conn.close()

def get_epochs_table_name(table_name):
    """
    For a table get the corresponding epochs
    table name
    """
    return '%s_epochs' % table_name

def get_band_cols():
    colnames=['nimage_tot',
              'nimage_use',
              'psf_flags',
              'psf_flux',
              'psf_flux_err',
              'psf_flux_s2n',
              'psf_mag',
              'psf_chi2per',
              'psf_dof',
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
    return ['tilename',
            'coadd_object_number',
            'flags',

            'psf_flags_g',
            'psf_flags_r',
            'psf_flags_i',
            'psf_flags_z',
            'psf_mag_g',
            'psf_mag_r',
            'psf_mag_i',
            'psf_mag_z',

            'exp_flags',
            'exp_chi2per',
            'exp_mag_g',
            'exp_mag_r',
            'exp_mag_i',
            'exp_mag_z',
            'exp_s2n_w',
            'exp_T_s2n',

            'dev_flags',
            'dev_chi2per',
            'dev_mag_g',
            'dev_mag_r',
            'dev_mag_i',
            'dev_mag_z',
            'dev_s2n_w',
            'dev_T_s2n']

def get_epoch_index_cols():
    return ['coadd_objects_id',
            'image_id',
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

    rc=deswl.files.Runconfig(run)
    releases=rc['dataset']
    bands=rc['band']

    info_list=desdb.files.get_coadd_info_by_release(releases,'g',
                                                    withbands=bands)

    tiles = [info['tilename'] for info in info_list]
    tiles=[tile for tile in tiles if tile not in _TILE_BLACKLIST]
    tiles.sort()

    return tiles


