from setuptools import setup, find_packages

import data_cleaning as module

setup(
    name='data-cleaning',
    version=module.__version__,
    author=module.__author__,
    author_email=module.team_email,
    description=module.package_info,
    license=module.package_license,
    packages=find_packages(),
)
