# Development Guide

## Getting started
In order to develop for this package:
1. Clone the github repository to your local computer.
2. Open project directory in your code editor using its graphical user interface and in your terminal using the the command `cd <project directory>`, where you need to replace `<project directory>` with your actual project directory.
3. Create a new empty conda environment for this project, e.g. using the Anaconda app or terminal command `conda create --name <environment name> python=<python version>`, where you need to replace `<environment name>` and `<python version>`. Then, activate the environment in your code editor using its graphical user interface and in your terminal using e.g. `conda activate <environment name>`.
4. Install all dependencies of the project's code using the terminal command `pip install --editable .`.
5. In your code editor, navigate to the source code (src directory) or unit tests (unit_tests) or documentation (docs) as you prefer. Whenever you install new dependencies, list them in the pyprojetc.toml file.

## Building a distributable version of the code
If you made any changes to the *c++ code*, do the following
1. Make sure you have a running c++ installation for you code editor.
2. Make sure you have a running installation of the cmake command toolbox (https://cmake.org/download/)
3. Then, in the terminal of your code editor, make sure your conda environment for this project is activated (see above)
4. Next, inside your terminal, install the pybind11 package using the command `pip install pybind11`
5. Then, make sure your CMakeLists.txt and pybind_wrapper.cpp files according to your requirements, since they will be used to compile your code and make it available to python.
6. Now, still inside your terminal, navigate to the folder where the c++ files are located that you edited and run the commands
   ```
   cmake -S. -Bbuild -Ax64
   cmake --build build -j
   ```
7. You should now be able to import your c++ functionality from within your python files.
8. Whenever you want to rebuild your c++ code, execute steps 5 to 7 again.

If you now want to build a distributable of the updated code (regardless of whether you updated the c++ or python code):
1. Install the required build tools using the command: `pip install build setuptools wheel`
2. Make sure the file *pyproject.toml* describes all specifications of your code properly. Typically, this involves incrementing the version number of your library and making sure that all new dependencies and packages of your code are properly listed. 
3. Delete the old distributables by removing the *.egg-info* located in the *src* directory as well as the *dist* directory which is located in the project's root directory. 
4. From within the project's root directory, build the new distributable according using the command: `python -m build`
   Observe how a new egg-info and a dist directory were created and filled with the distributable version of the project code.
5. Whenever you want to repeat the build, repeat steps 3 and 4.

## Verifying the validity of the distributbale
You can check whether your code is accessible and runs as expected as follows:
- Let the twine tool do some standard checks on your distributable. First, make sure twine is installed by running the terminal command `pip install twine` and then execute the test from within the project's root directory by using the command: `twine check dist/*`
- Do custom, manual checks by navigating to a different folder on your computer, creating a new python script where you test your new code. Then, create and activate a test conda environment and install the previosuly created distributable inside of it. This can be done using the command `pip install <path to distributable wheel>`, where `<path to distributable wheel>` is the path to the *.whl* file in the *dist* directory. Note that it might be necessary to put that path in quotation marks. In case you installed this wheel before, make sure to *un*install it before re-installing it, using `pip uninstall <path to distributbale wheel>`. Once you get back to the current project dircetory, make sure you switch back to the conda environemnt with which you develop this package. 

## Uploading the distributbale to PyPi
1. Make sure you have the api-key for uploading the package to PyPi or test PyPi. You can get them from the primary developer of this package.
2. Make sure you have twine installed by running the terminal command `pip install twine`
3. In order to upload the distributable to the test PyPi (recommended), run the terminal command `twine upload -r testpypi dist/*`. When it asks for the api key/ password, paste the key in. Note that on Windows, ctrl-v might not work. Use the Edit>Paste dialogue instead. Then, go on the test PyPi website and log in to see if the upload worked.
4. Assuming your distributable was uploaded successfully to test PyPi, you can upload it to PyPi by running the terminal command `twine upload dist/*`. Then, check on the PyPi website whether it worked successfully.

## Exporting the documentation to a local website
In order to export the documentation (mostly doc-comments inside the code) to a website that can be uploaded to read-the-docs:
1. Ensure you have installed sphinx using the terminal command `pip install sphinx sphinx_rtd_theme` .
2. In case no documentation website has been generated yet, 
   1. Create a folder called *docs* in the project's root directory.
   2. In the terminal, use the command `cd docs` to change its directory to the new docs folder.
   3. Run the terminal command `sphinx-quickstart` to generate some basic source documentation files. When asked, choose to separate the source from the build files such that we get two separate subfolders that are easier to work with later on. 
   4. Navigate to the file *conf.py* in docs/source and paste the below code snipppet in order to make sure your code (located in the *src* folder of this repository) will be found by sphinx. Note that this redirecting assumes that your terminal is currently in the docs folder and the .. makes sphinx go back to the project's root folder before going to the *src* folder.
      ```
      import os
      import sys
      sys.path.insert(0, os.path.abspath(os.path.join('..', '..', 'src')))
      ```
   5. Also include `extensions = ['sphinx.ext.autodoc', 'sphinx.ext.viewcode', 'sphinx.ext.todo']` and set `html_theme = 'sphinx_rtd_theme'`in the *conf.py* file. 
   6. In the file *index.rst* located in docs/source, make sure you have the following code
      ```
      Welcome to Project's documentation!
      ===================================

      .. toctree::
         :maxdepth: 2
         :caption: Contents:

         source/modules
               
      Indices and tables
      ==================

      * :ref:`genindex`
      * :ref:`modindex`
      * :ref:`search`

      ```
   4. Run the terminal command `sphinx-apidoc -o ./source ../src` which generates the sphinx source files based on the modules found in the *src* folder of our project. Since the terminal is currently inside the docs folder, we specify the location of the *src* folder using the ../ beforehand. The output files are the *.rst* files in the docs/source folder and there should be one for each of your python modules defined in the *src* folder. If one is missing, make sure you have an empty *__init__.py* file in each of your *src* packages, including *src* itself.
   5. Run the *python* terminal command `make html` to generate the build files from the source files. If you now open the *index.html* page from docs/build in your web-browser, you should see the documentation.
3. Then, whenever you want to export your updated documentation to a new build, delete the old *source folder* in docs. Then, make sure your terminal is in the *docs* folder (using the terminal command `cd docs`) and then run the terminal commands `sphinx-apidoc -o ./source ../src` and  `make html` as explained earlier. You can then inspect the website by opening the *index.html* file from the build/html folder in a web browser.
4. To make some further modifications to the webiste, you can for instance go the the file called *index.rst* and can write some information about the code. You can also adjust the design of the website by going to the file called *conf.py* inside the docs source folder and change the tag `html_theme` to `sphinx_rtd_theme` or any other theme you prefer. Make sure you enter your theme to the requirements.txt file in the docs/source folder such that the read-the-docs server installs it before building your website online. (see section on uploading documentation to read-the-docs).

## Upload the documentation to read-the-docs

