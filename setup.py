from setuptools import setup, find_packages

setup(
    name='chl-connect',
    use_scm_version=True,
    setup_requires=['setuptools-scm'],
    packages=find_packages(),
    install_requires=open('requirements.txt').read().splitlines(),
)