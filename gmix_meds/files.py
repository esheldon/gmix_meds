from __future__ import print_function
import os
import copy

DEFAULT_NPER=10

class Files(dict):
    """
    files for gmix meds fitting

    parameters
    ----------
    run: string
        The run identifier
    root_dir: optional, string
        Send a root directory for output.  If not sent, the default is gotten
        from the GMIX_MEDS_DATADIR environment variable
    """
    def __init__(self, run, root_dir=None):

        self._run=run
        self._set_root_dir(root_dir=root_dir)
        self._set_run_dir()

    def get_root_dir(self):
        """
        get the root directory
        """
        return self._root_dir

    def get_run_dir(self):
        """
        run directory

        parameters
        ----------
        run: string
            The run identifier
        """
        return self._run_dir

    def get_wq_dir(self, sub_dir=None):
        """
        wq directory

        parameters
        ----------
        sub_dir: string, optional
            An optional sub-directory to use, e.g.
            a tile name for DES data
        """

        if sub_dir is not None:
            return os.path.join(self._run_dir, 'wq', sub_dir)
        else:
            return os.path.join(self._run_dir, 'wq')

    def get_script_dir(self):
        """
        script directory
        """
        return os.path.join(self._run_dir,'script')

    def get_output_dir(self, sub_dir=None):
        """
        output directory

        parameters
        ----------
        sub_dir: string, optional
            An optional sub-directory to use, e.g.
            a tile name for DES data
        """
        if sub_dir is not None:
            return os.path.join(self._run_dir, 'output', sub_dir)
        else:
            return os.path.join(self._run_dir, 'output')

    def get_collated_dir(self):
        """
        collated directory
        """
        return os.path.join(self._run_dir, 'collated')


    def get_master_script_file(self):
        """
        get the path to the master script
        """
        sdir=self.get_script_dir()
        return os.path.join(sdir, 'master.sh')

    def get_output_file(self, split, sub_dir=None):
        """
        output directory file name

        parameters
        ----------
        split: 2-element sequence
            [beg,end] for split.
        sub_dir: string, optional
            An optional sub-directory name to use, e.g.
            a tile name for DES data

        %(run)s-%(beg)06d-%(end)%06d.fits
        %(run)s-%(sub_dir)s-%(beg)06d-%(end)%06d.fits
        """

        odir=self.get_output_dir(sub_dir=sub_dir)

        name=self._get_name(sub_dir=sub_dir, split=split, ext='fits')
        return os.path.join(odir, name)

    def get_collated_file(self, sub_dir=None, extra=None):
        """
        output directory file name

        parameters
        ----------
        sub_dir: string, optional
            An optional sub-directory name to use, e.g.
            a tile name for DES data
        extra: string
            Extra string, e.g. 'blind'

        %(run)s.fits
        %(run)s-%(sub_dir)s.fits
        %(run)s-%(sub_dir)s-%(extra)s.fits
        """

        odir=self.get_collated_dir()

        name=self._get_name(sub_dir=sub_dir, extra=extra, ext='fits')
        return os.path.join(odir, name)


    def get_wq_file(self, sub_dir=None, split=None):
        """
        output directory file name

        parameters
        ----------
        sub_dir: string, optional
            An optional sub-directory name to use, e.g.
            a tile name for DES data
        split: optional, 2-element sequence
            [beg,end] for split.
        """

        odir=self.get_wq_dir(sub_dir=sub_dir)

        name=self._get_name(sub_dir=sub_dir, split=split, ext='yaml')
        return os.path.join(odir, name)

    def _get_name(self, sub_dir=None, split=None, extra=None, ext='fits'):
        """
        get generic name, with sub_dir and split possibility
        """
        name=[copy.copy(self._run)]

        if sub_dir is not None:
            name.append(sub_dir)

        if split is not None:
            if len(split) != 2:
                raise ValueError("split should be [beg,end]")

            name.append('%06d' % split[0])
            name.append('%06d' % split[1])

        if extra is not None:
            name.append(extra)

        name='-'.join(name)
        name='%s.%s' % (name, ext)

        return name


    def _set_root_dir(self, root_dir=None):
        """
        set root dir, perhaps taking the default
        """
        if root_dir is None:
            root_dir=get_default_root_dir()

        root_dir=os.path.expandvars(root_dir)
        root_dir=os.path.expanduser(root_dir)
        self._root_dir=root_dir

    def _set_run_dir(self):
        """
        set the run directory
        """
        self._run_dir=os.path.join(self._root_dir, self._run)

def get_default_root_dir():
    """
    The root directory, under which the output run
    directories will be placed
    """
    root_dir=os.environ['GMIX_MEDS_DATADIR']
    return root_dir

def get_chunks(ntot, nper):
    """
    get a chunklist 

    [ [beg1,end1], [beg2,end2], ...]

    """
    indices=numpy.arange(nobj)
    nchunk=ntot/nper

    chunk_list=[]

    for i in xrange(nchunk):

        beg=i*nper
        end=(i+1)*nper
        
        if end > (ntot-1):
            end=ntot-1

        chunklist.append( [beg,end] )

    return chunklist

def get_condor_head_template():
    text="""
Universe        = vanilla

Notification    = Never 

# Run this exe with these args
Executable      = %(master_script)s

Image_Size      = 1000000

GetEnv = True

kill_sig        = SIGINT

requirements = (cpu_experiment == "star") || (cpu_experiment == "phenix")
#requirements = (cpu_experiment == "star")

+Experiment     = "astro"\n\n"""
    return text

def get_wq_template():
    text="""
command: |
    source ~/.bashrc
    module unload gmix_meds && module load gmix_meds/work

    config_file="%(config_file)s"
    meds_files="%(meds_files_csv)s"
    beg="%(beg)s"
    end="%(end)s"
    out_file="%(out_file)s"
    log_file="%(log_file)s"

    master_script=%(master_script)s

    $master_script $config_file $meds_files $beg $end $out_file $log_file

job_name: %(job_name)s\n"""

    return text

def get_master_script_text():
    text="""#!/bin/bash
function go {
    hostname

    python -u $GMIX_MEDS_DIR/bin/gmix-fit-meds     \\
            --obj-range $beg,$end                  \\
            --work-dir $tmpdir                     \\
            $config_file $meds_file $out_file
    
    exit_status=$?

}

if [ $# -lt 5 ]; then
    echo "error: config_file meds_file beg end out_file"
    exit 1
fi

# this can be a list
config_file="$1"
meds_file="$2"
beg="$3"
end="$4"
out_file="$5"
log_file="$6"

if [[ -n $_CONDOR_SCRATCH_DIR ]]; then
    tmpdir=$_CONDOR_SCRATCH_DIR
else
    tmpdir=$TMPDIR
    mkdir -p $tmpdir
fi

outdir=$(dirname $out_file)
mkdir -p $outdir

lbase=$(basename $log_file)
tmplog="$tmpdir/$lbase"

go &> "$tmplog"
mv -v "$tmplog" "$log_file"

exit $exit_status

    \n"""

    return text


class StagedOutFile(object):
    """
    A class to represent a staged file

    If tmpdir=None no staging is performed and the original file
    path is used

    parameters
    ----------
    fname: string
        Final destination path for file
    tmpdir: string, optional
        If not sent, or None, the final path is used and no staging
        is performed
    must_exist: bool, optional
        If True, the file to be staged must exist at the time of staging
        or an IOError is thrown. If False, this is silently ignored.
        Default False.

    examples
    --------

    # using a context for the staged file
    fname="/home/jill/output.dat"
    tmpdir="/tmp"
    with StagedOutFile(fname,tmpdir=tmpdir) as sf:
        with open(sf.path,'w') as fobj:
            fobj.write("some data")

    # without using a context for the staged file
    sf=StagedOutFile(fname,tmpdir=tmpdir)
    with open(sf.path,'w') as fobj:
        fobj.write("some data")
    sf.stage_out()

    """
    def __init__(self, fname, tmpdir=None, must_exist=False):
        self.final_path=fname
        self.tmpdir=tmpdir
        self.must_exist=must_exist

        self.was_staged_out=False

        if tmpdir is None:
            self.is_temp=False
            self.path=self.final_path
        else:
            self.is_temp=True

            if not os.path.exists(tmpdir):
                os.makedirs(tmpdir)

            bname=os.path.basename(fname)
            self.path=os.path.join(tmpdir, bname)

            if self.path == self.final_path:
                # the user sent tmpdir as the final output dir, no
                # staging is performed
                self.is_temp=False

    def stage_out(self):
        """
        if a tempdir was used, move the file to its final destination

        note you normally would not call this yourself, but rather use a
        context, in which case this method is called for you

        with StagedOutFile(fname,tmpdir=tmpdir) as sf:
            #do something
        """
        import shutil

        if self.is_temp and not self.was_staged_out:
            if not os.path.exists(self.path):
                if self.must_exist:
                    raise IOError("temporary file not found:",self.path)
            else:
                if os.path.exists(self.final_path):
                    print("removing existing file:",self.final_path)
                    os.remove(self.final_path)

                makedir_fromfile(self.final_path)
                print("staging out",self.path,"->",self.final_path)
                shutil.move(self.path,self.final_path)

        self.was_staged_out=True

    def __enter__(self):
        return self
    def __exit__(self, exception_type, exception_value, traceback):
        self.stage_out()

def makedir_fromfile(fname):
    dname=os.path.dirname(fname)
    if not os.path.exists(dname):
        try:
            os.makedirs(dname)
        except:
            # probably a race condition
            pass


