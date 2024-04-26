import os
from setuptools import setup, find_packages


def get_requirements(fns):
    reqs = []
    for fn in fns:
        if not os.path.exists(fn):
            raise FileNotFoundError(f'Given file {fn} does not exists.')
        with open(fn, 'r') as f:
            reqs += [line.strip() for line in f.readlines()]
    return reqs


setup(
    name='dcmtrans',
    description='Library for reading various of dicom image(s).',
    setuptools_git_versioning={
        'enabled': True,
        'template': '{tag}',
        'dev_template': '{tag}.post{ccount}',
        'dirty_template': '{tag}.post{ccount}+dirty',
    },
    extras_require={
        'all': [
            'nibabel>=4.0.1',
            'matplotlib',
        ],
    },
    setup_requires=['setuptools-git-versioning'],
    install_requires=get_requirements(['requirements.txt']),
    packages=find_packages(exclude=['test']),
)

