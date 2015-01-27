#!/usr/bin/env python
import numpy

def _get_nbrs(mindex,m,buff_frac=0.25,maxsize=512):
    """
    Gets nbrs of any postage stamp in the MEDS.
    
    A nbr is defined as any stamp which overlaps the stamp under consideration.
    Stamps with size 256 are doubled to 512.    
    """
    
    #check that current gal has OK stamp, or return bad crap
    if m['orig_start_row'][mindex,0] == -9999 or m['orig_start_col'][mindex,0] == -9999:
        nbr_numbers = numpy.array([-1],dtype=int)
        return nbr_numbers        
    
    #expand the stamps and get edges
    dsize = (maxsize-256)/2
    sze = m['box_size'].copy()
    l = m['orig_start_row'][:,0].copy()
    r = m['orig_start_row'][:,0].copy()
    b = m['orig_start_col'][:,0].copy()
    t = m['orig_start_col'][:,0].copy()
    
    q, = numpy.where(sze == 256)
    if q.size > 0:
        sze[q[:]] = maxsize
        l[q[:]] -= dsize
        b[q[:]] -= dsize
        
    r += sze
    t += sze

    #get the nbrs from two sources
    # 1) intersection of postage stamps
    # 2) seg map vals
    nbr_numbers = []
    
    #box intersection test and exclude yourself
    #use buffer of 1/4 of smaller of pair of stamps
    buff = sze.copy()
    q, = numpy.where(buff[mindex] < buff)
    if len(q) > 0:
        buff[q[:]] = buff[mindex]
    buff = buff*buff_frac
    q, = numpy.where((~((l[mindex] > r-buff) | (r[mindex] < l+buff) | (t[mindex] < b+buff) | (b[mindex] > t-buff))) & (m['number'][mindex] != m['number']))
    if len(q) > 0:
        nbr_numbers.extend(list(q.copy()+1))

    #check coadd seg maps
    try:
        segmap = m.get_cutout(mindex,0,type='seg')
        q = numpy.where((segmap > 0) & (segmap != mindex+1))
        if len(q) > 0:
            nbr_numbers.extend(list(numpy.unique(segmap[q])))    
    except:
        pass
    
    #cut weird crap
    if len(nbr_numbers) > 0:
        nbr_numbers = numpy.array(nbr_numbers,dtype=int)
        nbr_numbers = numpy.unique(nbr_numbers)
        inds = nbr_numbers-1
        q, = numpy.where((m['orig_start_row'][inds,0] != -9999) & (m['orig_start_col'][inds,0] != -9999))
        if len(q) > 0:
            nbr_numbers = list(nbr_numbers[q])
        else:
            nbr_numbers = []
    
    #if have stuff return unique else return -1
    if len(nbr_numbers) == 0:
        nbr_numbers = numpy.array([-1],dtype=int)
    else:
        nbr_numbers = numpy.array(nbr_numbers,dtype=int)
        nbr_numbers = numpy.unique(nbr_numbers)
    
    return nbr_numbers

def get_meds_nbrs(meds_list,conf):
    """
    For a given list of MEDS files, returns all the nbrs of every object.
    
    The returned array has two columns
    
        number - the number in the seg map of the central object
        nbr_number - the number of the nbr
    
    To get all of the nbrs of object num, one just needs to find all rows 
    with the same number field like this
    
        q, = numpy.where(nbrs_data['number'] == num)
        nbrs_numbers = nbrs_data['nbr_number'][q]
    """
    #data types
    nbrs_data = []
    dtype = [('number','i8'),('nbr_number','i8')]

    #loop through objects, get nbrs in each meds list
    for mindex in xrange(meds_list[0].size):
        nbrs = []
        for m in meds_list:
            #make sure MEDS lists have the same objects!
            assert m['number'][mindex] == meds_list[0]['number'][mindex]
            assert m['id'][mindex] == meds_list[0]['id'][mindex]

            #add on the nbrs
            nbrs.extend(list(_get_nbrs(mindex,m,buff_frac=float(conf['overlap_frac']),maxsize=int(conf['size_for_256']))))

        #only keep unique nbrs
        nbrs = numpy.unique(numpy.array(nbrs))
        
        #add to final list
        for nbr in nbrs:
            nbrs_data.append((mindex+1,nbr))
    
    #return array sorted by number
    nbrs_data = numpy.array(nbrs_data,dtype=dtype)
    i = numpy.argsort(nbrs_data['number'])
    nbrs_data = nbrs_data[i]

    return nbrs_data

class NbrsFoF(object):
    def __init__(self,nbrs_data):
        self.nbrs_data = nbrs_data
        self.Nobj = len(numpy.unique(nbrs_data['number']))

        #what if we percolate?
        sys.setrecursionlimit(self.Nobj)
        
        #holds links of all current groups
        self.links = numpy.zeros(self.Nobj,dtype='i8')

        #records in fof group has been processed
        # -1 == not linked yet, will be tested right away
        # >= 0 : fof index (or id)
        self.linked = numpy.zeros(self.Nobj,dtype='i8')

        #holds first index, num and last index of fof groups
        self.fofs_head = []
        self.fofs_num = []
        self.fofs_tail = []

        self.fof_data = None
        
        #make fofs on init
        self.make_fofs()

    def write_fofs(self,fname):
        fitsio.write(fname,self.fof_data,clobber=True)
            
    def _make_fof_data(self):
        self.fof_data = []
        for i in xrange(self.Nobj):
            self.fof_data.append((self.linked[i],i+1))
        self.fof_data = numpy.array(self.fof_data,dtype=[('fofid','i8'),('number','i8')])
        i = numpy.argsort(self.fof_data['number'])
        self.fof_data = self.fof_data[i]
        assert numpy.all(self.fof_data['fofid'] >= 0)
        
    def _init_fofs(self):
        self.links[:] = -1
        self.linked[:] = -1
        self.fofs_head = []
        self.fofs_num = []
        self.fofs_tail = []

    def _get_nbrs_index(self,mind):
        q, = numpy.where((self.nbrs_data['number'] == mind+1) & (self.nbrs_data['nbr_number'] > 0))
        if len(q) > 0:
            return list(self.nbrs_data['nbr_number'][q]-1)
        else:
            return []

    def _recursive_link_fof(self,mind,fofind):
        #get nbrs for this object
        nbrs = set(self._get_nbrs_index(mind))
        
        #add links to tail of fof
        for ids in list(nbrs):
            if self.linked[ids] == -1:
                self.links[self.fofs_tail[fofind]] = copy.copy(ids)
                self.fofs_tail[fofind] = copy.copy(ids)
                self.fofs_num[fofind] += 1
                self.linked[ids] = fofind
                self._recursive_link_fof(ids,fofind)

    def make_fofs(self):
        #init
        self._init_fofs()

        #link
        import progressbar
        bar = progressbar.ProgressBar(maxval=self.Nobj,widgets=[progressbar.Bar(marker='|', left='doing work: |', right=''), ' ', progressbar.Percentage(), ' ', progressbar.AdaptiveETA()])
        bar.start()
        for i in xrange(self.Nobj):
            bar.update(i+1)
            if self.linked[i] == -1:
                self.fofs_head.append(i)
                self.fofs_tail.append(i)
                self.fofs_num.append(1)            
                fofind = len(self.fofs_head)-1            
                self.linked[i] = fofind
                self._recursive_link_fof(i,fofind)
        bar.finish()
        
        self._make_fof_data()


def NbrsFoFExtractor(object):
    """
    Class to extract subet set of FoF file and destroy on exit if wanted.
    """

    def __init__(self, fof_file, start, end, sub_file, cleanup=False):
        self.fof_file = fof_file
        self.start = start
        self.end = end
        self.sub_file = sub_file
        self.cleanup = cleanup
        self._check_inputs()
        
        self._extract()

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()

    def __del__(self):
        self.close()

    def close(self):
        if self.cleanup:
            if os.path.exists(self.sub_file):
                print 'removing sub file:',self.sub_file
                os.remove(self.sub_file)

    def _get_inds(self, data):
        inds = []
        for fofid in range(self.start,self.end+1):
            q, = numpy.where(data['fofid'] == fofid)
            if len(q) > 0:
                inds.extend(list(q))
        inds = numpy.array(inds,dtype=int)
        return inds
                    
    def _extract(self):
        
        with fitsio.FITS(self.meds_file) as infits:
            print 'opening sub file:',self.sub_file
            with fitsio.FITS(self.sub_file,'rw',clobber=True) as outfits:
                old_data = infits[1][:]
                inds = self._get_inds(old_data)
                obj_data = old_data[inds]
                outfits.write(obj_data)

    def _check_inputs(self):
        if self.meds_file==self.sub_file:
            raise ValueError("output file name equals input")

        if self.start > self.end:
            raise ValueError("one must extract at least one object")

