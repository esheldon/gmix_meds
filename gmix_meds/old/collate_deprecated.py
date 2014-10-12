#
# deprecated
#

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
    Concatenate the split files
    
    This version designed for DES where the sub-dir is a tilename and
    run info is declared in deswl.  Moving away from this.
    """
    def __init__(self, run, tilename, ftype='mcmc', blind=True, clobber=False):
        import desdb
        import deswl

        self.run=run
        self.tilename=tilename
        self.ftype=ftype
        self.blind=blind
        self.clobber=clobber

        self.rc=deswl.files.read_runconfig(run)
        self.bands=self.rc['bands']
        self.nbands=len(self.bands)
        self.config = load_config(self.rc['config'])

        self.nper=self.rc['nper']

        self.df=desdb.files.DESFiles()

        self.set_collated_file()
        self.find_coadd_run()

        self.load_meds()
        self.set_nrows()


        if self.blind:
            self.blind_factor = get_blind_factor()

    def load_meds(self):
        """
        Load the associated meds files
        """
        import meds
        print('loading meds')

        self.meds_list=[]
        for band in self.bands:
            fname=self.df.url('meds',
                              tilename=self.tilename,
                              band=band,
                              medsconf=self.rc['medsconf'],
                              coadd_run=self.coadd_run)
            m=meds.MEDS(fname)
            self.meds_list.append(m)

    def find_band_image_ids(self, medsobj, conn):

        # use hash to get unique run-exp combos
        runexp_dict={}

        # keyed by run-exp-ccd holding index into _image_ids
        image_indexes={}

        nimage=medsobj._image_info.size
        iids = numpy.zeros(nimage, dtype='i8')

        # first get all the run-exposure combinations
        for index in xrange(1,nimage):
            fname=medsobj._image_info['image_path'][index]
            ii=extract_red_info(fname)

            rexp='%(run)s-%(expname)s' % ii
            rexpccd='%s-%02d' % (rexp, ii['ccd'])

            ii['rexp'] = rexp
            ii['rexpccd'] = rexpccd

            image_indexes[rexpccd] = index

            runexp_dict[rexp] = ii


        # now go through each run-exposure combo, pull out info
        # for all ccds, and then match against our input

        found=0
        for rexp,ii in runexp_dict.iteritems():
            query="""
            select
                id,run,exposurename as expname,ccd
            from
                location
            where
                run='%(run)s'
                and exposurename='%(expname)s'
                and filetype='red'""" % ii

            res=conn.quick(query)
            for r in res:
                rexpccd='%(run)s-%(expname)s-%(ccd)02d' % r

                index=image_indexes.get(rexpccd,None)
                if index is not None:
                    iids[index] = r['id']
                    found += 1

        if found != nimage-1:
            raise RuntimeError("expected to find %d but "
                               "found %d" % (nimage-1,found))

        return iids

    def cache_allband_se_image_ids(self):
        """
        cache for all bands
        """
        import desdb
        conn=desdb.Connection()
        for band in xrange(self.nbands):

            self.cache_se_image_ids(band, conn=conn)
        conn.close()

    def cache_se_image_ids(self, band, conn=None):
        """
        Get the image id from the database for each of the 
        SE exposures.  Cache on disk
        """
        import desdb

        self.ensure_cache_dir_exists()

        print('cacheing image ids for band:',band)
        if conn is None:
            conn_use=desdb.Connection()
        else:
            conn_use=conn

        cache_name=self.get_se_band_cache_filename(band)

        m=self.meds_list[band]

        iids=self.find_band_image_ids(m, conn_use)

        print(cache_name)
        fitsio.write(cache_name,iids, extname="image_ids", clobber=True)

        if conn is None:
            conn_use.close()

    def read_se_image_ids(self, band):
        """
        read from the image id cache
        """
        cache_name=self.get_se_band_cache_filename(band)
        if not os.path.exists(cache_name):
            raise ConcatError("cache file does not exist: %s" % cache_name)
            #self.cache_se_image_ids(band)
        print("reading:",cache_name)
        return fitsio.read(cache_name) 

    def set_se_image_ids(self):
        """
        set the image ids on the meds object for the requested band
        """
        for band in xrange(self.nbands):
            se_ids = self.read_se_image_ids(band)
            self.meds_list[band]._image_ids = se_ids

    def find_coadd_run(self):
        """
        Get the coadd run from tilename and medsconf
        """
        import glob
        meds_dir = self.df.dir('meds_run',medsconf=self.rc['medsconf'])
        pattern = os.path.join(meds_dir,'*%s*' % self.tilename)

        print(pattern)

        flist=glob.glob(pattern)
        if len(flist) != 1:
            raise ConcatError("expected 1 dir, found %d" % len(flist))

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

    def get_cache_dir(self):
        cache_dir = self.df.dir('wlpipe_collated', run=self.run)
        cache_dir=os.path.join(cache_dir,'cache')
        return cache_dir

    def ensure_cache_dir_exists(self):
        cache_dir=self.get_cache_dir()
        if not os.path.exists(cache_dir):
            print("making cache dir:",cache_dir)
            try:
                os.makedirs(cache_dir)
            except:
                # probably a race condition
                pass



    def get_se_band_cache_filename(self, band):
        cache_dir=self.get_cache_dir()
        fname='%s-se-image-id-cache-%d.fits' % (self.tilename,band)
        path=os.path.join(cache_dir, fname)
        return path

    def get_coadd_cache_filename(self):
        cache_dir=self.get_cache_dir()
        fname='%s-coadd-image-info.fits' % self.tilename
        path=os.path.join(cache_dir, fname)
        return path


    def set_collated_file(self):

        out_dir = self.df.dir('wlpipe_collated', run=self.run)
        if not os.path.exists(out_dir):
            try:
                os.makedirs(out_dir)
            except:
                # probably a race condition
                pass

        if self.blind:
            out_ftype='wlpipe_me_collated_blinded'
        else:
            out_ftype='wlpipe_me_collated'
            
        self.collated_file = self.df.url(out_ftype,
                                         run=self.run,
                                         tilename=self.tilename,
                                         filetype=self.ftype,
                                         ext='fits')

        bname=os.path.basename(self.collated_file)
        if '_CONDOR_SCRATCH_DIR' in os.environ:
            temp_dir=os.environ['_CONDOR_SCRATCH_DIR']
        else:
            temp_dir=os.environ['TMPDIR']
        self.temp_file=os.path.join(temp_dir, bname)

    '''
    def do_setup(self):
        self.set_se_image_ids()
        self.set_coadd_objects_info()
    '''
    def verify(self):
        """
        just run through and read the data, verifying we
        can read it and that it matches to the coadd
        """
        from deswl.generic import get_chunks, extract_start_end

        #self.do_setup()

        nper=self.rc['nper']
        startlist,endlist=get_chunks(self.nrows,nper)
        nchunk=len(startlist)

        dlist=[]
        elist=[]

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

            print('\t%d/%d %s' %(i+1,nchunk,fname))
            try:
                data, epoch_data = self.read_data(fname,start,end)
            except ConcatError as err:
                print("error found: %s" % str(err))


    def concat(self):
        import esutil as eu
        import shutil
        from deswl.generic import get_chunks, extract_start_end
        print('will write:',self.collated_file)

        #self.do_setup()

        nper=self.rc['nper']
        startlist,endlist=get_chunks(self.nrows,nper)
        nchunk=len(startlist)

        if os.path.exists(self.collated_file) and not self.clobber:
            print('file already exists, skipping')
            return

        dlist=[]
        elist=[]

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

            if ((i+1) % 100) == 0:
                print('\t%d/%d %s' %(i+1,nchunk,fname))
            data, epoch_data = self.read_data(fname,start,end)

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

        print("writing temp file:",self.temp_file)
        with fitsio.FITS(self.temp_file,'rw',clobber=True) as fobj:
            meta=fitsio.read(fname, ext="meta_data")
            fobj.write(data,extname="model_fits")

            if do_epochs > 0:
                fobj.write(epoch_data,extname='epoch_data')

            fobj.write(meta,extname="meta_data")

        print("moving temp file:",self.temp_file,self.collated_file)
        shutil.move(self.temp_file,self.collated_file)
        print('output is in:',self.collated_file)


    def blind_data(self,data):
        """
        multiply all shear type values by the blinding factor

        This also includes the Q values from B&A
        """
        from .lmfit import get_model_names
        simple_models=self.config.get('simple_models',lmfit.SIMPLE_MODELS_DEFAULT )

        names=data.dtype.names
        for model in simple_models:
            n=get_model_names(model)

            g_name=n['g']
            e_name=n['e']
            Q_name=n['Q']
            flag_name=n['flags']

            if flag_name in names:
                w,=numpy.where(data[flag_name] == 0)
                if w.size > 0:
                    if g_name in names:
                        data[g_name][w,:] *= self.blind_factor

                    if e_name in names:
                        data[e_name][w,:] *= self.blind_factor

                    if Q_name in names:
                        data[Q_name][w,:] *= self.blind_factor

    def pick_epoch_fields(self, epoch_data0):
        """
        pick out some fields, add some fields, rename some fields
        """
        import esutil
        #name_map={'number':'coadd_object_number', # to match the database
        #          'psf_fit_g':'psf_fit_e'}
        name_map={'psf_fit_g':'psf_fit_e'}
        rename_columns(epoch_data0, name_map)

        # we can loosen this when we store the cutout sub-id
        # in the output file....  right now file_id can be
        # not set
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
        #dt =  [('coadd_objects_id','i8'),('image_id','i8')] + dt

        epoch_data = numpy.zeros(epoch_data0.size, dtype=dt)
        esutil.numpy_util.copy_fields(epoch_data0, epoch_data)

        for band_num in xrange(self.nbands):
            w,=numpy.where(epoch_data['band_num'] == band_num)
            if w.size > 0:

                epoch_data['band'][w] = self.bands[band_num]

                #m=self.meds_list[band_num]
                #file_ids=epoch_data['file_id'][w]
                #epoch_data['image_id'][w] = m._image_ids[file_ids]

        return epoch_data

    def pick_fields(self, data0, meta):
        """
        pick out some fields, add some fields, rename some fields
        """
        import esutil

        nbands=self.nbands
        name_map={'number':     'coadd_object_number',

                  'coadd_exp_g':      'coadd_exp_e',
                  'coadd_exp_g_cov':  'coadd_exp_e_cov',
                  'coadd_exp_g_sens': 'coadd_exp_e_sens',
                  'coadd_dev_g':      'coadd_dev_e',
                  'coadd_dev_g_cov':  'coadd_dev_e_cov',
                  'coadd_dev_g_sens': 'coadd_dev_e_sens',

                  'exp_g':      'exp_e',
                  'exp_g_cov':  'exp_e_cov',
                  'exp_g_sens': 'exp_e_sens',
                  'dev_g':      'dev_e',
                  'dev_g_cov':  'dev_e_cov',
                  'dev_g_sens': 'dev_e_sens'}
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
        names = ['coadd_objects_id','tilename'] + names

        
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
        esutil.numpy_util.copy_fields(data0, data)
        data['tilename'] = self.tilename


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



    def add_coadd_objects_id(self, data):
        """
        match by coadd_object_number and add the coadd_objects_id

        if we have a corrupted file, this will raise ConcatError
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
            for i in xrange(data.size):
                print(data['coadd_object_number'], data['coadd_objects_id'])
            raise ConcatError("only %d/%d matched by "
                              "coadd_object_number" % (nmatch,data.size))

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



    def read_data(self, fname, start, end):
        """
        Read the chunk data
        """
        if not os.path.exists(fname):
            raise ConcatError("file not found: %s" % fname)

        try:
            with fitsio.FITS(fname) as fobj:
                data0       = fobj['model_fits'][:]
                epoch_data0 = fobj['epoch_data'][:]
                meta        = fobj['meta_data'][:]
        except IOError as err:
            raise ConcatError(str(err))

        expected_index = numpy.arange(start,end+1)+1
        w,=numpy.where(data0['number'] != expected_index)
        if w.size > 0:
            raise ConcatError("number field is corrupted in file: %s" % fname)

        data = self.pick_fields(data0,meta)
        self.add_coadd_objects_id(data)

        if epoch_data0.dtype.names is not None:
            epoch_data = self.pick_epoch_fields(epoch_data0)
            if epoch_data.dtype.names is not None:
                self.add_coadd_objects_id(epoch_data)
        else:
            epoch_data = epoch_data0

            
        return data, epoch_data

    '''
    def cache_coadd_objects_info(self):
        """
        cache the coadd info on disk
        """
        import desdb
        print('cacheing coadd_objects ids')

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

        self.ensure_cache_dir_exists()

        print('getting coadd objects info')
        conn=desdb.Connection()
        res=conn.quick(query, array=True)
        conn.close()

        cache_name=self.get_coadd_cache_filename()
        print(cache_name)
        fitsio.write(cache_name,res,extname="coadd_info",clobber=True)

    def read_coadd_objects_info(self):
        """
        read the coadd info cache
        """
        cache_name=self.get_coadd_cache_filename()
        if not os.path.exists(cache_name):
            raise ConcatError("cache file does not exist: %s" % cache_name)
            #self.cache_coadd_objects_info()

        print("reading:",cache_name)
        res=fitsio.read(cache_name,extname="coadd_info")
        return res


    def set_coadd_objects_info(self):
        """
        set the coadd info 
        """
        res=self.read_coadd_objects_info()

        nmax=res['coadd_object_number'].max()
        if nmax != res.size:
            raise ConcatError("some missing object_number, got "
                              "max %d for nobj %d" % (nmax,res.size))


        self.coadd_objects_data=res
        self.min_object_number=res['coadd_object_number'].min()
        self.max_object_number=res['coadd_object_number'].max()
    '''

    def read_meds_meta(self, meds_files):
        """
        get the meds metadata
        """
        pass


