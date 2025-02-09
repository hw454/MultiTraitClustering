"""
Author: Hayley Wragg
Created: 6th February 2025
Description:
    This module sets up the `multitraitclustering` package using setuptools.
    The `multitraitclustering` package applies clustering methods to GWAS data to identify pathways. 
    The clustering methods handle association data across many phenotype traits, and the output 
    will also return the pathway enrichment for the clusters.
    Input type: `filename.csv` should contain the association scores between SNPs and phenotypes.
    Each row corresponds to a SNP and each column corresponds to a trait. The first column is the 
    SNP rsid labels. The first row is the trait labels.
Attributes:
    name (str): The name of the package.
    version (str): The current version of the package.
    author (str): The author of the package.
    author_email (str): The email of the author.
    description (str): A short description of the package.
    long_description (str): A detailed description of the package.
    long_description_content_type (str): The content type of the long description.
    packages (list): list all Python import packages that should be included in the distribution.
    classifiers (list): A list of classifiers to categorize the project.
    python_requires (str): The required Python version for the package.
"""

from setuptools import setup, find_packages

LONG_DESCRIPTION = """
Applies clustering methods to GWAS data to identify separate pathway groups.
The clustering methods handle association data across many phenotype traits.
The output will also return the pathway enrichment for the clusters.

Input type: `filename.csv` should contain the association scores between SNPs
and phenotypes. Each row corresponds to a SNP and each column corresponds to
a trait. The first column is the SNP rsid labels. The first row is the trait
labels."""

setup(
    name="multitraitclustering",
    version="0.1.36",
    author="Hayley Wragg",
    author_email="hayleywragg@hotmail.com",
    description="Applies clustering to GWAS data to identify pathways",
    long_description = LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
