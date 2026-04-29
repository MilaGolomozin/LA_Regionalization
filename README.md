# LA Regionalization Pipeline

This repository provides a pipeline for the systematic regionalization of the left atrial (LA) surface. The approach builds upon the work of Marta Nunez Garcia, with the key distinction that seed points for region definition are computed automatically, removing the need for manual user input.

## Overview

The pipeline is structured into sequential processing steps, each implemented in a dedicated module:

### 1. Pulmonary Vein (PV) Separation and Labeling
Automatic identification and segmentation of the pulmonary veins, followed by assignment of anatomical labels.

### 2. Mitral Valve Contour Detection
Extraction and labeling of the mitral valve (MV) contour.

### 3. Seed Extraction and Regionalization
Identification of nine anatomical seed points and computation of geodesic paths between them to define atrial regions.

## Current Limitations

- The current implementation is designed specifically for anatomies with **four pulmonary veins (4 PV configuration)**.
- File paths are **hard-coded**, limiting usability across multiple datasets or batch processing scenarios.
- The pipeline has been **tested on the [Public Cardiac CT Dataset](https://github.com/Bjonze/Public-Cardiac-CT-Dataset?tab=readme-ov-file)**, but has not yet been generalized to other datasets or anatomical variations.

## Future Work

- Generalize the pipeline to support **variable pulmonary vein anatomies**.
- Improve automation to allow processing of **multiple scans or entire directories**.
- Remove hard-coded paths and introduce configurable input/output handling.

## References

This work is based on the methodology presented in:

> Nunez Garcia, M. (2018). *Left atrial parameterisation and multi-modal data analysis: application to atrial fibrillation.*

Related implementation by the original author:  
https://github.com/martanunez/LA_flattening
