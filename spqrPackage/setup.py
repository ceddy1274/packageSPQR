from setuptools import setup, find_packages

with open("app/spqr/README.md", "r") as f:
    long_description = f.read()

setup(
    name='spqr',
    version='1.5',
    description='Package for implementation of Semi Parametric Quantile Regression (SPQR)',
    package_dir={"": "app"},
    packages=find_packages(where="app"),
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Cameron Eddy',
    author_email='ceddy1274@gmail.com',
    license='GPL-3.0',
    license_files=['LICENSE'],
    classifiers=[
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
    ],
    install_requires=[
       'torch',
       'numpy',
       'pandas',
       'scipy',
       'matplotlib',
       'scikit-learn',
    ],
)