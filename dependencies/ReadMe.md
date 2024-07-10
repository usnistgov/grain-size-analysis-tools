# Overview of Installation Instructions

The Grain Size Library will run on Windows, OS X, and Linux environments. It was developed in Python 3.10 using the Miniforge package manager, which is similar to Anaconda. Using the files provided in this directory, the necessary dependencies can be installed using either the Anaconda, the Miniforge, or the PIP package manager. Note, the Anaconda package manager uses the "conda" command, whereas Miniforge utilizes identical commands but substitutes the "conda" command for "mamba". More information can be found here:

[Anaconda documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)

[Miniforge GitHub](https://github.com/conda-forge/miniforge)


## Installing Dependencies Using Miniforge/Mamba

This directory contains .yml text files which are used to define an enviroment within the Anaconda or Miniforge package managers. Before the dependencies can be installed using Mamba, the Miniforge Python command prompt must first be installed. The latest installers for Miniforge3 can be downloaded and installed from the [Miniforge GitHub](https://github.com/conda-forge/miniforge) repository. 

After installation, begin a new command prompt (or terminal) with the Miniforge environment activated. A new Python environment named "metal_env" will be created from the dependencies outlined in the provided .yml file. To do this, type the following the command prompt,

  `mamba env create -f anaconda_env_py310_generic.yml`

Although the specific dependencies need to be installed only once, this newly created Python environment must be activated each time a new Miniforge command prompt is opened. To do so, type the following,

  `mamba activate metal_env`

With the new environment activated, the Python examples provided in this library can now be run by typing the following,

  `python path_to_script.py`

where `path_to_script.py` should be replaced with the script that will be executed.


## Installing Dependencies Using Anaconda/Conda

The process for installing the dependencies within Anaconda are nearly identical to the steps taken above for Miniforge. First, Anaconda must be downloaded installed, which is described on the [Anaconda website](https://docs.anaconda.com/free/anaconda/install/index.html).

After installation, open up a new command prompt (or terminal) with the Anaconda environment activated. Then follow the steps above described using Miniforge, but replace the `mamba` commands with `conda` commands.


## Installing Dependencies Using PIP

As an alternative to Mamba or Conda, the dependencies can also be installed using PIP. First, install Python 3.10 and PIP on your operating system. This can be easily done in Anaconda or Miniforge using the "anaconda_env_py310_generic_pip.yml" environment file. If you used this environment (.yml) file, be sure to first activate the new environment called "metal_env". 

`mamba env create -f anaconda_env_py310_generic_pip.yml`

`mamba activate metal_env`

Otherwise, if you are not using Anaconda/Miniforge, it is recommended that you install the necessary dependencies in a Python [virtual environment](https://docs.python.org/3.10/library/venv.html).

Next, use the provided "requirements_py310.txt" file to automatically install all of the dependencies at once using PIP,

  `pip install --user --prefer-binary -r requirements_py310.txt`
