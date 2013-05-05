import meds
import gmix_image

class MedsLM(object):
    def __init__(self, meds_file, det_cat=None):
        self._meds_file=meds_file
        self._meds=meds.MEDS(meds_file)

        meta=self._meds.get_meta()

    def process_object(self, index):
        """
        Process the indicated object

        The first cutout is always the coadd, followed by
        the SE images which will be fit simultaneously
        """
        pass
