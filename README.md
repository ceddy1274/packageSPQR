# SPQR: Semi-Parametric Quantile Regression <img src="figures/logo.png" align="right" height="120" alt="" />
The full code is provided in the spqrPackage folder. You can also download the current version of the package at https://test.pypi.org/project/SPQR/ (if need be you may need to update the version to the most current one as opposed to the version below). Make sure you install the torch vision audio and dms variants package before installing the SPQR package. An example can be seen below:
```
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip install dms_variants
!pip install -i https://test.pypi.org/simple/ SPQR==1.5
```
The .png figures in the direction have been uploaded so they can be displayed in the PyPi package. Without them in the GitHub, we would not be able to display the images in the PyPi package.

If you would like to create a new SPQR package, you must first download the spqrPackage (you need to download this whole folder and should run all commands in the spqrPackage directory). We used this YouTube video as a reference for how to upload the package: https://www.youtube.com/watch?v=5KEObONUkik.

Once you have the package downloaded it is likely you will need to change some files. For the most part, you should only ever need to change: main.py, README.md, or setup.py. If you are changing the code of the package itself, you should update main.py. If you would like to update documentation, then you should update README.md. Finally, there could be a time where you need to update setup.py. This may occur if you do something such as change the license of the package, need to add a new import, or need to debug something while uploading to testpypi.org or pypi.org.

Once you have completed updates to the package run these commands to upload it to testpypi.org:
```
python setup.py bdist_wheel sdist
twine upload -r testpypi dist/* --verbose
```
Or these commands to upload it to pypi.org:
```
python setup.py bdist_wheel sdist
twine upload dist/* --verbose
```
It is likely you will be asked for an API token when you run these commands. Therefore, you will need to make a PyPi account and generate an API token for the package. Then you can copy this API token in the place it asks (note API tokens are private, so it will not show you what you have typed). It may asks for a login instead of a API token. If that is the case, use a login instead of an API token. 
