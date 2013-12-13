
def make_tile_input(run, tilename, ftype, blind=True, create=False):
    """
    Create the control file and potentially the create table file
    """
    import desdb
    import deswl
    import fitsio
    if blind:
        out_ftype='wlpipe_me_collated_blinded'
    else:
        out_ftype='wlpipe_me_collated'


    rc=deswl.files.Runconfig(run)
    bands=rc['band']
    band_cols=get_band_cols()

    df=desdb.files.DESFiles()
    fname = df.url(out_ftype,
                   run=run,
                   tilename=tilename,
                   filetype=ftype,
                   ext='fits')

    with fitsio.FITS(fname) as fobj:
        model_fits=fobj['model_fits'][:]
        epoch_data=fobj['epoch_data'][:]


    control_file=fname.replace('.fits','.ctl')
    epoch_control_file=fname.replace('.fits','-epoch.ctl')

    table, epoch_table = get_table_names(run)

    desdb.array2table(model_fits, table, control_file,
                      bands=bands, band_cols=band_cols,
                      create=create,
                      primary_key='coadd_objects_id')
    desdb.array2table(epoch_data, epoch_table, epoch_control_file,
                      create=create)

def add_indexes(run):
    """
    Add indexes to the appropriate columns
    """
    import desdb
    table, epoch_table = get_table_names(run)

    index_cols=get_index_cols()
    epoch_index_cols=get_epoch_index_cols()

    qt="create index {col}_idx on {table}({col})"

    conn=desdb.Connection()
    curs = conn.cursor()
    for col in index_cols:
        query = qt.format(table=table, col=col)
        print query

        curs.execute(query)

    for col in epoch_index_cols:
        query = qt.format(table=epoch_table, col=col)
        print query

        curs.execute(query)

    curs.close()
    conn.close()

def get_table_names(run):
    """
    Table names based on run
    """
    table=run
    epoch_table='%s_epochs' % run
    return table, epoch_table

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
