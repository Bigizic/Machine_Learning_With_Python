``In this notebook, you will use SVM (Support Vector Machines) to build and train a model using human cell records, and classify cells to whether the samples are benign or malignant.``

``SVM works by mapping data to a high-dimensional feature space so that data points can be categorized, even when the data are not otherwise linearly separable. A separator between the categories is found, then the data is transformed in such a way that the separator could be drawn as a hyperplane. Following this, characteristics of new data can be used to predict the group to which a new record should belong.``

The example is based on a dataset that is publicly available from the UCI

The dataset consists of several hundred human cell sample records, each of which contains the values of a set of cell characteristics. The fields in each record are:

| Field name | Description |
| ---- | ---- |
| ID | Clump thickness |
| Clump | Clump thickness |
| UnitSize | Uniformity of cell size |
| UnitShape | Uniformity of cell shape |
| MargAdh | Marginal adhension |
| SingEpiSize | Single epithelial cell size |
| BareNuc | Bare nuclei |
| BlandChrom | Bland chromatin |
| NormNucl | Normal nucleoil |
| Mit | Mitoses |
| Class | Benign or malignant |
| ----- | ----- |
