from setuptools import find_packages, setup

with open("READme.md", "r",) as f:
    long_description = f.read()

setup(
    name="pybriann",
    version="0.0.1",
    description="",
    package_dir={"":"src"},
    packages=find_packages(where="src"),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TimHenry1995/pybriann",
    author="",
    author_email="",
    license="MIT",
    classifiers=[
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.10",
    "Operating System :: OS Independent"
    ],
    install_requires=["numpy >= 2.2.5"],
    extras_require={
    "dev": ["pytest>=7.0", "twine>=6.1.0", "wheel >= 0.45.1", "setuptools >= 79.0.0"],
    },
    python_requires=">=3.13.2"
)