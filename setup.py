import setuptools
import subprocess
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

classifiers = [
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Operating System :: POSIX",
    "Operating System :: MacOS :: MacOS X",
    "Operating System :: Microsoft :: Windows",
]


def is_cuda_available():
    try:
        output = subprocess.check_output(['nvcc', '--version'])
        return 'release' in output.decode('utf-8')
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


install_requires=[
    "setuptools",
    "numpy==1.26.0",
    "pandas==2.1.1",
    "nltk==3.8.1",
    "anytree==2.10.0",
    "networkx==3.1",
    "chardet==5.2.0",
    "jellyfish==1.0.1",
    "PuLP==2.7.0",
    "pyemd==1.0.0",
    "python-dateutil==2.8.2",
    "pytest~=8.2.0",
    "torch==2.2.1",
    "scipy~=1.11.3",
    "scikit-learn~=1.3.1",
    "transformers==4.37.2",
    "datasets>=2.9.0",
    "sentence-transformers",
    "tqdm~=4.66.1",
    "plotly",
    "umap-learn",
]

if is_cuda_available():
    install_requires.append("cupy-cuda11x")
    install_requires.append("pylibraft-cu11")
    install_requires.append("POT @ git+https://github.com/PythonOT/POT.git")
else:
    install_requires.append("POT")

setuptools.setup(
    name='valentwin',
    version='0.0.0-alpha.1',
    description='Valentwin Matcher',
    classifiers=classifiers,
    license_files=('LICENSE',),
    author='Albertus Andito',
    author_email='a.andito@sussex.ac.uk',
    maintainer='Albertus Andito',
    maintainer_email='a.andito@sussex.ac.uk',
    # url='https://delftdata.github.io/valentine/',
    # download_url='https://github.com/delftdata/valentine/archive/refs/tags/v0.1.8.tar.gz',
    packages=setuptools.find_packages(exclude=('tests*', 'examples*')),
    dependency_links=["https://pypi.nvidia.com"],
    install_requires=install_requires,
    keywords=['matching', 'valentine', 'schema matching', 'dataset discovery', 'coma', 'cupid', 'similarity flooding'],
    include_package_data=True,
    python_requires='>=3.8,<3.13',
    long_description=long_description,
    long_description_content_type='text/markdown'
)
