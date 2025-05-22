# Development Guide

## Getting started
In order to develop for this package:
1. Clone the github repository to your local computer.
2. Open project directory in your code editor using its graphical user interface and in your terminal using the the command `cd <project directory>`, where you need to replace `<project directory>` with your actual project directory.
3. Create a new empty conda environment for this project, e.g. using the Anaconda app or terminal command `conda create --name <environment name> python=<python version>`, where you need to replace `<environment name>` and `<python version>`. Then, activate the environment in your code editor using its graphical user interface and in your terminal using e.g. `conda activate <environment name>`.
4. Install all dependencies of the project's code using the terminal command `pip install --editable .`.
5. In your code editor, navigate to the source code (src directory) or unit tests (unit_tests) or documentation (docs) as you prefer. Whenever you install new dependencies, list them in the pyprojetc.toml file.

## Developing the code

### C++
- You can find the c++ code of this project in the *src/briann/c_plus_plus* directory as well as a file called *CMakeLists.txt* (that helps with c++ compilation) in the project's root directory. 
- In order to make your c++ code accessible inside your python code, the library [pybind11](https://pybind11.readthedocs.io/en/stable/index.html) is used. It allows you to create python modules that bind to your c++ functions. The two most important steps are to build a new module via pybind11's [*PYBIND11_MODULE*](https://pybind11.readthedocs.io/en/stable/reference.html) command (see below) and to configure its compilation via the *CMakeLists.txt*. In order to make sure pybind11 can bind your c++ functions to your python modules, you need to adhere to the following rules.
   - *Changing the folder structure:* The basic folder structure of this project stores all code in an *src* folder with a *briann* folder inside. Then, there is a c_plus_plus folder collecting all c++ code of the project in several subfolders. It is important that all folders that should be recognized as packages by python and by the sphinx documentation tool (see corresponding section) contain an *\_\_init\_\_* file that is typically empty. Each subfolder represents a package, i.e. a set of files (also known as modules) that provide a certain functionality. For each module (think of a typical python file), there are two files here, namely a *.cpp* file and a *.py* file, both having the same name (see below for why this is useful). It is recommended to align any changes in the file layout (such as renaming folders, moving modules around, adding modules, etc.) to this overall setup. In case you want to change the name of the briann folder, make sure you change the name of the project in the *pyproject.toml* file (located in the project's root directory) to the same value. Otherwise, the build tool (scikit-build, specified in pyproject.toml) will not be able to bind the modules generated from your c++ code to the project.
   - *Adding a new c++ function to an existing module:* You can write a new c++ function, e.g. next to an existing c++ function inside an existing .cpp file. Then, inside that .cpp file, inspect the current setup of the [*PYBIND11_MODULE*](https://pybind11.readthedocs.io/en/stable/reference.html) macro. This is a command that tells pybind11 how it should set up a python module containing the entry points to the c++ functions you want to make available to python. Make sure you register your new c++ function with this method. 
   - *Adding a new module to an existing package:* Every module is here represented with two files, namely a *.cpp* and a *.py* file of same name. The *.cpp* file defines the c++ functionality and configures the mapping from c++ to a corresponding python module that is generated automatically with the help of the [pybind11](https://pybind11.readthedocs.io/en/stable/index.html). To create a new module, create an empty file (the module) with *.cpp* extension in the same style as the existing modules inside an exisiting package (the encapsulating folder). This means, with the same `#include` statements at the top and the same general setup for the [*PYBIND11_MODULE*](https://pybind11.readthedocs.io/en/stable/reference.html) macro. This macro allows you specify the module name and the documentation for your functions. Now, navigate to the file *CMakeLists.txt* (in the project's root directory) to configure how the compiler shall bind your c++ module to the python module. You should use the commands `python_add_library`, `target_link_libraries`, `target_compile_definitions` and `install` similar to how they are being used for the already configured modules. Note that the `install` command acts like placing your new module inside a package, such that in your python code you can import it as `from <package name> import <module name>`, where you need to replace `<package name>` and `<module name>` with your chosen names. In python, your imports will thus be using the namespace of your *compiled packages and modules* instead of the folder and file structure you see in the left hand file-browser pane of your code editor. While you can import the functionality using this generated module's name inside your own python scripts (e.g. in the briann/python folder), that generated python module will be hard to find for the sphinx documentation tool. This is why a small *.py* file of same name as the *.cpp* is used here. It simply imports all the functionality from the generated python module and stores it in its *\_\_all\_\_* special variable. The sphinx documentation tool will then look for this *.py* file that you declare manually (rather than the one that was automatically generated by pybind) and take the documentation from that *\_\_all\_\_* variable. 
   - *Adding a new package to the existing c_plus_cplus package:* Create a new folder (the new package) inside the *src/c_plus_plus* folder. Make sure you have an empty file in there called *\_\_init\_\_.py*, or otherwise your folder will not be considered as a package by the scikit-build. Then, place some modules inside as described above. 

Note that you cannot use this c++ code inside of your own python code before having generated a wheel from it and installed it locally. To do both in one go, navigate with your temrinal to the project's root directory (using the `cd` command) and then run the command `pip install .` to generate a temporary wheel of your own code and install it. To prevent having to rebuild your entire project every time you made a change to your c++ code, consider writing unit tests in c++ before the build.

### Python
You can make changes to the python code found in the *src/briann/python* folder. 
- Make sure that every folder has an empty *\_\_init\_\_.py* file to make it recocnizable as a package to python and the documentation tool sphinx. 
- If you want to use c++ code in your python code, see the section on editing c++ code, to make sure you understand how that code is set up. 
- Remember, whenever you make use of a new external library, add it as a dependency in the *pyproject.toml* file found at the project's root directory.

## Building a distributable version of the code

The distributable (typically a .whl file) is version of your code that can be run by the users of your project. Use the following steps to build a distributable.

1. In the terminal of your code editor, make sure your conda environment for this project is activated (see above)
2. Make sure you have scikit-build installed by running the command `pip install scikit-build`.
3. Make sure your code (both c++ and python) runs bug-free and that the *CmakeLists.txt* and *pyproject.toml* files are configured properly, since they configure the build.
4. In case you have a folder called *dist* in the project's root directory, make sure to delete any old builds (the *.whl* as well as the *.tar.gz* files) that you no longer need. This prevents interference during installation of your distributable later on.
5. Inside your terminal, make sure you are in the project's root direcory (using the `cd` command as needed) and use the command `python -m build`. It is possible that it throws a warning about intermediate directories and incremental builds. You can usually ignore this warning. Then, observe how a *dist* folder was created (in case it did not yet exist) and that it contains a new *.whl* and *.tar.gz* file. You will typically only need the *.whl* file.

## Verifying the validity of the distributable
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
   1. Install your own code by making sure your terminal is pointing to the project's root directory and the using the command `pip install .`.
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

         modules
               
      Indices and tables
      ==================

      * :ref:`genindex`
      * :ref:`modindex`
      * :ref:`search`

      ```
   4. Run the terminal command `sphinx-apidoc -o ./source ../src/briann` which generates the sphinx source files based on the modules found in the *src* folder of our project. Since the terminal is currently inside the docs folder, we specify the location of the *src* folder using the ../ beforehand. The output files are the *.rst* files in the docs/source folder and there should be one for each of your python modules defined in the *src* folder. If one is missing, make sure you have an empty *__init__.py* file in each of your *src* packages, including *src* itself.
   5. Run the *python* terminal command `make html` to generate the build files from the source files. If you now open the *index.html* page from docs/build in your web-browser, you should see the documentation.
3. Then, whenever you want to export your updated documentation to a new build, delete the *source folder* in docs. This is important because the old files might not be overwritten if they already exist and the files that are no longer needed are not going to be deleted automatically by the next sphinx command, yet, leaving these old files present will likley lead to inteference later on. Next, review the *conf.py* file inside the docs folder and make sure the cofniguration is according to your liking. Pay attention to the release number which you might want to keep in sync with the version number provided in your *pyproject.toml* file. Then, make sure your terminal is in the project's root folder and reinstall your code with the command `pip install .`. Next, navigate your terminal to the *docs* folder (using the command `cd docs`) and then run the commands `sphinx-apidoc -o ./source ../src/briann` and  `make html` as explained earlier. You can then inspect the website by opening the *index.html* file from the build/html folder in a web browser. Note, you might want to prevent having to reinstall your package every time you want to update your documentation. However, since here we are using scikit-build (rather that setuptools) as our build engine, this is difficult. At the time of this writing, editable installs are still experimental and thus not recommended.
4. To make some further modifications to the webiste, you can for instance go the the file called *index.rst* and can write some information about the code. You can also adjust the design of the website by going to the file called *conf.py* inside the docs source folder and change the tag `html_theme` to `sphinx_rtd_theme` or any other theme you prefer. Make sure you enter your theme to the requirements.txt file in the docs/source folder such that the read-the-docs server installs it before building your website online. (see section on uploading documentation to read-the-docs).

## Upload the documentation to read-the-docs

