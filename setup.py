from setuptools import setup, find_packages

long_description = """
Applies clustering methods to GWAS data to identify separate pathway groups.
The clustering methods handle association data across many phenotype traits.
The output will also return the pathway enrichment for the clusters.

Input type: `filename.csv` should contain the association scores between SNPs
and phenotypes. Each row corresponds to a SNP and each column corresponds to
a trait. The first column is the SNP rsid labels. The first row is the trait
labels."""

setup(
    name="multitraitclustering",
    version="0.1.22",
    author="Hayley Wragg",
    author_email="hayleywragg@hotmail.com",
    description="Applies clustering to GWAS data to identify pathways",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
