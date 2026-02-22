"""
Setup script for ENS-GI Digital Twin package
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_file(filename):
    filepath = os.path.join(os.path.dirname(__file__), filename)
    # Try encodings in order: utf-8-sig (UTF-8 BOM), utf-16 (Windows BOM), utf-8, latin-1
    for enc in ('utf-8-sig', 'utf-16', 'utf-8', 'latin-1'):
        try:
            with open(filepath, encoding=enc) as f:
                return f.read()
        except (UnicodeDecodeError, LookupError):
            continue
    return ''

setup(
    name='ens-gi-digital-twin',
    version='0.3.0',
    author='Mahdad',
    author_email='your.email@example.com',
    description='Multiscale Digital Twin for Enteric Nervous System and GI Motility',
    url='https://github.com/yourusername/ens-gi-digital-twin',
    license='MIT',

    # Package discovery from src/
    packages=find_packages(where="src"),
    package_dir={"": "src"},

    python_requires='>=3.8',

    install_requires=[
        'numpy>=1.21.0',
        'scipy>=1.7.0',
        'matplotlib>=3.5.0',
        'seaborn>=0.11.0',
        'pandas>=1.3.0',
        'tqdm>=4.62.0',
    ],

    extras_require={
        'pinn': [
            'tensorflow>=2.15.0',
            'tensorflow-probability>=0.22.0',
        ],
        'bayesian': [
            'pymc>=5.10.0',  # Modern PyMC (v5+)
            'arviz>=0.12.0',
            'pytensor>=2.18.0',  # Modern backend
        ],
        'optimization': [
            'numba>=0.55.0',
            'tqdm>=4.62.0',
        ],
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=3.0.0',
            'black>=22.0.0',
            'flake8>=4.0.0',
            'sphinx>=4.5.0',
            'sphinx-rtd-theme>=1.0.0',
        ],
        'all': [
            'tensorflow>=2.15.0',
            'tensorflow-probability>=0.22.0',
            'pymc>=5.10.0',  # Modern PyMC (v5+)
            'arviz>=0.12.0',
            'pytensor>=2.18.0',  # Modern backend
            'numba>=0.55.0',
            'tqdm>=4.62.0',
            'pytest>=7.0.0',
            'pytest-cov>=3.0.0',
            'black>=22.0.0',
            'flake8>=4.0.0',
            'sphinx>=4.5.0',
            'sphinx-rtd-theme>=1.0.0',
        ],
    },

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Healthcare Industry',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
        'Topic :: Scientific/Engineering :: Physics',
    ],

    keywords='digital-twin neuroscience gastrointestinal IBS PINN bayesian-inference neuromorphic',

    project_urls={
        'Documentation': 'https://ens-gi-digital-twin.readthedocs.io/',
        'Source': 'https://github.com/yourusername/ens-gi-digital-twin',
        'Tracker': 'https://github.com/yourusername/ens-gi-digital-twin/issues',
    },
)
