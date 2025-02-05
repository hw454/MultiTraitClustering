Requirements:
    python=3.11.9 (not currently compatible with higher versions)
    scipy
    scikit-learn
    scikit-learn-extra
    numpy
    pandas

Setup:
    Installation:
        ```pip install multitraitclustering```
    Data:
        * Association Data. This is your master data set with association scores corresponding
        to a collection of GWAS analysis. The columns correspond to traits, the rows and snps and the values are the association between the snp and the trait. This should be stored as a `.csv` file with the trait labels on the first row and the snps on the first column. 
        * Exposure Data. This is a dataset corresponding to only the exposure phenotype and the association with the snps. Also stored as a csv file, rows are snps, one column for the exposure labelled `EXP` and the values are the scores.
        * P-values and Standard Error values are not yet incorporated into the analysis.

Usage Example:
    ```
