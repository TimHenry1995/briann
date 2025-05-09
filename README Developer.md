# Development Guide

## Getting started
In order to develop for this package:
1. Create a new empty conda environment for this project and activate it
2. Continue the development of the source code (src directory) or unit tests (unit_tests) or documentation (docs) as you prefer.

## Building a distributable version of the code
In order to distribute the updated code:
1. Install the required build tools using the command: `pip install build setuptools wheel`
2. Make sure the file *pyproject.toml* describes all specifications of your code properly. Typically, this involves incrementing the version number of your library and making sure that all new dependencies and packages of your code are properly listed. 
3. Delete the old distributables by removing the *.egg-info* located in the *src* directory as well as the *dist* directory which is located in the project's root directory. 
3. From within the project's root directory, build the new distributable according using the command: `python -m build`
   Observe how a new egg-info and a dist directory were created and filled with the distributable version of the project code.

## Verifying the validity of the distributbale
You can check whether your code is accessible and runs as expected as follows:
- Let the twine tool do some standard checks on your distributable. First, make sure twine is installed by running: `pip install twine` and then execute the test from within the project's root directory by using the command: `twine check dist/*`
- Do custom, manual checks by creating and activating a test conda environment and installing the distributable inside of it. This can be done using the command `pip install <path to distributable wheel>`, where `<path to distributable wheel>` is the path to the *.whl* file in the *dist* directory. Note that it might be necessary to put that path in quotation marks. In case you installed this wheel before, make sure to *un*install it before re-installing it, using `pip uninstall <path to distributbale wheel>`. Then, go to a different folder on your computer and create a python script in which you use the new functionality. After actviating the new test environment, you should be able to run that python script. Once you get back to the current projetc dircetory, make sure you switch back to the conda environemnt with which you develop this package. 

## Uploading the distributbale to PyPi
1. Make sure you have the api-key for uploading the package to PyPi or test PyPi. You can get them from the primary developer of this package.
2. Make sure you have twine installed by running `pip install twine`
3. In order to upload the distributable to the test PyPi (recommended), run the command `twine upload -r testpypi dist/*`. When it asks for the api key/ password, paste the key in. Note that on Windows, ctrl-v might not work. Use the Edit>Paste dialogue instead. Then, go on the test PyPi website and log in to see if the upload worked.
4. Assuming your distributable was uploaded successfully to test PyPi, you can upload it to PyPi by running `twine upload dist/*`. Then, check on the PyPi website whether it worked successfully.
