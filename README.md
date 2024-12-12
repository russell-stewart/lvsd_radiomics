# Radiomics in the Etiologic Subtyping of Left Ventricular Systolic Dysfunction
Russell Stewart BA BE, Ryan Samarakoon MS, Eric Farber-Eger MS, Ravinder Mallugari MBBS, and Quinn Wells MD PharmD MSCI

This repository serves as a digital supplement to the above-titled poster presentation.

## Table of Contents
- `features/*`: Supplemental tables of selected radiomic features chosen through RFECV
- `requirements.txt`: Python package requirements for this project (Python 3.9.2)
- `ValidateMasksExtractFeatures.ipynb`: Verify quailty of T1 map segmentation outputs, and extract features using `pyradiomics` with project-specific parameters
- `radiomic_extraction_parameters.yaml`: `pyradiomics`-formatted feature extraction parameters
- `FitAndValidateModel.ipynb`: Train and perform internal validation to classify heart failure etiology
- `HFClassification.py`: Helper enumeration used to define and group heart failure etiologies
- `FeatureSelector.py`: Helper sklearn transformer with combined feature selection via RFECV and heirarchical clustering used in the classifier notebook
- `model.joblib`: The fit model used in the poster presentation

## References
1. Bloom MW, Greenberg B, Jaarsma T, et al. Heart failure with reduced ejection fraction. Nat Rev Dis Primers. 2017;3(1):17058. [doi:10.1038/nrdp.2017.58](doi:10.1038/nrdp.2017.58)
2. Sara JD, Toya T, Taher R, Lerman A, Gersh B, Anavekar NS. Asymptomatic Left Ventricle Systolic Dysfunction. Eur Cardiol. 2020;15:e13. [doi:10.15420/ecr.2019.14](doi:10.15420/ecr.2019.14)
3. Lee E, Ibrahim ESH, Parwani P, Bhave N, Stojanovska J. Practical Guide to Evaluating Myocardial Disease by Cardiac MRI. American Journal of Roentgenology. 2020;214(3):546-556. [doi:10.2214/AJR.19.22076](doi:10.2214/AJR.19.22076)
4. Mayerhoefer ME, Materka A, Langs G, et al. Introduction to Radiomics. J Nucl Med. 2020;61(4):488-495. [doi:10.2967/jnumed.118.222893](doi:10.2967/jnumed.118.222893)
5. Wang J, Yang F, Liu W, et al. Radiomic Analysis of Native T1 Mapping Images Discriminates Between MYH7 and MYBPC3-Related Hypertrophic Cardiomyopathy. J Magn Reson Imaging. 2020;52(6):1714-1721. [doi:10.1002/jmri.27209](doi:10.1002/jmri.27209)
6. Wang ZC, Fan ZZ, Liu XY, et al. Deep Learning for Discrimination of Hypertrophic Cardiomyopathy and Hypertensive Heart Disease on MRI Native T1 Maps. J Magn Reson Imaging. 2024;59(3):837-848. [doi:10.1002/jmri.28904](doi:10.1002/jmri.28904)
7. Wang J, Bravo L, Zhang J, et al. Radiomics Analysis Derived From LGE-MRI Predict Sudden Cardiac Death in Participants With Hypertrophic Cardiomyopathy. Front Cardiovasc Med. 2021;8:766287. [doi:10.3389/fcvm.2021.766287](doi:10.3389/fcvm.2021.766287)
8. Zhang H, Zhao L, Wang H, et al. Radiomics from Cardiovascular MR Cine Images for Identifying Patients with Hypertrophic Cardiomyopathy at High Risk for Heart Failure. Radiol Cardiothorac Imaging. 2024;6(1):e230323. [doi:10.1148/ryct.230323](doi:10.1148/ryct.230323)
9. Zhang J, Xu Y, Li W, et al. The Predictive Value of Myocardial Native T1 Mapping Radiomics in Dilated Cardiomyopathy: A Study in a Chinese Population. J Magn Reson Imaging. 2023;58(3):772-779. [doi:10.1002/jmri.28527](doi:10.1002/jmri.28527)
10. Petersen SE, Matthews PM, Francis JM, et al. UK Biobank’s cardiovascular magnetic resonance protocol. Journal of Cardiovascular Magnetic Resonance. 2016;18(1):8. [doi:10.1186/s12968-016-0227-4](doi:10.1186/s12968-016-0227-4)
11. Bai W, Sinclair M, Tarroni G, et al. Automated cardiovascular magnetic resonance image analysis with fully convolutional networks. Journal of Cardiovascular Magnetic Resonance. 2018;20(1):65. [doi:10.1186/s12968-018-0471-x](doi:10.1186/s12968-018-0471-x)
12. Ma J, He Y, Li F, Han L, You C, Wang B. Segment anything in medical images. Nat Commun. 2024;15(1):654. [doi:10.1038/s41467-024-44824-z](doi:10.1038/s41467-024-44824-z)
13. van Griethuysen JJM, Fedorov A, Parmar C, et al. Computational Radiomics System to Decode the Radiographic Phenotype. Cancer Research. 2017;77(21):e104-e107. [doi:10.1158/0008-5472.CAN-17-0339](doi:10.1158/0008-5472.CAN-17-0339)