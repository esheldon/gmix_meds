import os
import copy

def get_default_root_dir():
    """
    The root directory, under which the output run
    directories will be placed
    """
    root_dir=os.environ['GMIX_MEDS_DATADIR']
    return root_dir

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

        self._run_run
        if root_dir is None:
            self._root_dir=get_default_root_dir()
        else:
            self._root_dir=root_dir

        self._run_dir=os.path.join(self._root_dir, run)

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

    def get_script_dir(self, run):
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

    def get_output_file(self, sub_dir=None, split=None):
        """
        output directory file name

        parameters
        ----------
        sub_dir: string, optional
            An optional sub-directory name to use, e.g.
            a tile name for DES data
        split: optional, 2-element sequence
            [beg,end] for split.

        %(run)s.fits
        %(run)s-%(sub_dir)s.fits
        %(run)s-%(beg)06d-%(end)%06d.fits
        %(run)s-%(sub_dir)s-%(beg)06d-%(end)%06d.fits
        """

        odir=self.get_output_dir(sub_dir=sub_dir)

        name=[copy.copy(self._run)]

        if sub_dir is not None:
            name.append(sub_dir)

        if split is not None:
            if len(split) != 2:
                raise ValueError("split should be [beg,end]")

            name.append('%06d' % split[0])
            name.append('%06d' % split[1])

        name='-'.join(name)
        name='%s.fits' % name

        return os.path.join(odir, name)
