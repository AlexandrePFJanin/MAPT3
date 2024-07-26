from setuptools import setup

setup(
   name='MAPT3',
   version='0.1.0',
   author='Alexandre JANIN',
   author_email='alexandre.janin@protonmail.com',
   url='https://github.com/AlexandrePFJanin/MAPT3',
   packages=['MAPT3'],
   license='LICENSE.md,
   description='Multi-disciplinary and Automatic Plate Tessellation and Time tracking Toolkit.',
   long_description=open('README.md').read(),
   install_requires=[
        'ipython>=8.15.0',
        'numpy>=1.12',
        'matplotlib>=3.0',
	    'cartopy>=0.18',
	    'scipy>=1.5.2',
	    'tqdm>=4.65.0',
	    'alphashape>=1.3.1',
	    'shapely>=2.0.2',
	    'h5py>=3.9.0'
   ],
)
