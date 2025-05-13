# Development Guide

## Getting started
In order to develop for this package:
1. Clone the github repository to your local computer.
2. Open project directory in your code editor using its graphical user interface and in your terminal using the the command `cd <project directory>`, where you need to replace `<project directory>` with your actual project directory.
3. Create a new empty conda environment for this project, e.g. using the Anaconda app or terminal command `conda create --name <environment name> python=<python version>`, where you need to replace `<environment name>` and `<python version>`. Then, activate the environment in your code editor using its graphical user interface and in your terminal using e.g. `conda activate <environment name>`.
4. Install all dependencies of the project's code using the terminal command `pip install --editable .`.
5. In your code editor, navigate to the source code (src directory) or unit tests (unit_tests) or documentation (docs) as you prefer. Whenever you install new dependencies, list them in the pyprojetc.toml file.

## Building a distributable version of the code
In order to distribute the updated code:
1. Install the required build tools using the command: `pip install build setuptools wheel`
2. Make sure the file *pyproject.toml* describes all specifications of your code properly. Typically, this involves incrementing the version number of your library and making sure that all new dependencies and packages of your code are properly listed. 
3. Delete the old distributables by removing the *.egg-info* located in the *src* directory as well as the *dist* directory which is located in the project's root directory. 
3. From within the project's root directory, build the new distributable according using the command: `python -m build`
   Observe how a new egg-info and a dist directory were created and filled with the distributable version of the project code.

## Verifying the validity of the distributbale
You can check whether your code is accessible and runs as expected as follows:
- Let the twine tool do some standard checks on your distributable. First, make sure twine is installed by running the terminal command `pip install twine` and then execute the test from within the project's root directory by using the command: `twine check dist/*`
- Do custom, manual checks by navigating to a different folder on your computer, creating a new python script where you test your new code. Then, create and activate a test conda environment and install the previosuly created distributable inside of it. This can be done using the command `pip install <path to distributable wheel>`, where `<path to distributable wheel>` is the path to the *.whl* file in the *dist* directory. Note that it might be necessary to put that path in quotation marks. In case you installed this wheel before, make sure to *un*install it before re-installing it, using `pip uninstall <path to distributbale wheel>`. Once you get back to the current project dircetory, make sure you switch back to the conda environemnt with which you develop this package. 

## Uploading the distributbale to PyPi
1. Make sure you have the api-key for uploading the package to PyPi or test PyPi. You can get them from the primary developer of this package.
2. Make sure you have twine installed by running the terminal command `pip install twine`
3. In order to upload the distributable to the test PyPi (recommended), run the terminal command `twine upload -r testpypi dist/*`. When it asks for the api key/ password, paste the key in. Note that on Windows, ctrl-v might not work. Use the Edit>Paste dialogue instead. Then, go on the test PyPi website and log in to see if the upload worked.
4. Assuming your distributable was uploaded successfully to test PyPi, you can upload it to PyPi by running the terminal command `twine upload dist/*`. Then, check on the PyPi website whether it worked successfully.
