import os
import glob
from distutils.core import setup

scripts=['gmix-fit-meds',
         'gmix-meds-collate',
         'gmix-fit-double-psf',
         'gmix-fit-meds-stars',
         'gmix-meds-make-scripts', # des specific
         'gmix-meds-make-oracle']  # des specific


scripts=[os.path.join('bin',s) for s in scripts]

conf_files=glob.glob('config/*.yaml')
runconf_files=glob.glob('runconfig/*.yaml')

data_files=[]
for f in conf_files:
    d=os.path.dirname(f)
    n=os.path.basename(f)
    d=os.path.join('share','gmix-meds-config')

    data_files.append( (d,[f]) )

for f in runconf_files:
    d=os.path.dirname(f)
    n=os.path.basename(f)
    d=os.path.join('share','gmix-meds-runconfig')

    data_files.append( (d,[f]) )



setup(name="gmix_meds", 
      version="0.1.0",
      description="Run gaussian mixtures on MEDS files",
      license = "GPL",
      author="Erin Scott Sheldon",
      author_email="erin.sheldon@gmail.com",
      scripts=scripts,
      data_files=data_files,
      packages=['gmix_meds'])
