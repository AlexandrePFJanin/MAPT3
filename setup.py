from setuptools import setup

setup(
    name='MAPT3',
    version='0.1.2',
    author='Alexandre JANIN',
    author_email='alexandre.janin@protonmail.com',
    url='https://github.com/AlexandrePFJanin/MAPT3',
    description='Multi-disciplinary and Automatic Plate Tessellation and Time tracking Toolkit.',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    license='Apache License 2.0',
    packages=['MAPT3'],
    include_package_data=True,
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
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)
