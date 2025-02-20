# Sea-Ice and Lead Unsupervised Classification

This project implements machine learning algorithms to differentiate between sea ice and lead using satellite imagery. The classification is based on Sentinel-3 altimetry data and is validated against ESA's ground truth classification. The project explores unsupervised classification methods, specifically K-means clustering and Gaussian Mixture Models (GMM), comparing their strengths and weaknesses.

## Table of Contents
- [Introduction to Unsupervised Learning](#introduction-to-unsupervised-learning)
  - [K-means Clustering](#k-means-clustering)
  - [Gaussian Mixture Models (GMM)](#gaussian-mixture-models-gmm)
- [Getting Started](#getting-started)
  - [Data Sources](#data-sources)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)

## Introduction to Unsupervised Learning

### K-means Clustering
K-means is an unsupervised learning algorithm that partitions a dataset into k clusters. It assigns data points to the nearest cluster centroid and iteratively updates centroids to minimize within-cluster variation.

#### Why K-means for Clustering?
- No prior knowledge about data distribution required.
- Simple and scalable for large datasets.

#### Key Components of K-means
- **Choosing K**: The number of clusters is predefined.
- **Centroid Initialization**: Affects final clustering.
- **Assignment Step**: Assigns each point to the nearest centroid.
- **Update Step**: Updates centroids based on assigned points.

#### Advantages of K-means
- Computational efficiency.
- Easily interpretable clustering results.

### Gaussian Mixture Models (GMM)
GMM assumes the dataset comprises multiple Gaussian distributions, allowing soft clustering by estimating the probability of each data point belonging to different clusters.

#### Why Gaussian Mixture Models for Clustering?
- Provides probabilistic clustering instead of hard assignment.
- More flexible covariance structure, allowing clusters to take various shapes.

#### Key Components of GMM
- **Number of Components**: Similar to K in K-means, defines the number of Gaussian distributions.
- **Expectation-Maximization (EM) Algorithm**: Iteratively updates model parameters to maximize likelihood.
- **Covariance Type**: Determines the flexibility of cluster shapes.

#### Advantages of GMM
- Suitable for datasets with overlapping clusters.
- Captures complex distributions better than K-means.

## Getting Started

This project is implemented in **Google Colab**, leveraging cloud-based resources for computation.


### Data Sources
The dataset consists of Sentinel-2 optical data and Sentinel-3 OLCI data. These data files can be retrieved from the **Copernicus Data Space** following the steps outlined in [Data Fetching](https://dataspace.copernicus.eu/).

**Sentinel-2 Optical Data:**
- `S2A_MSIL1C_20190301T235611_N0207_R116_T01WCU_20190302T014622.SAFE`

**Sentinel-3 OLCI Data:**
- `S3B_SR_2_LAN_SI_20190301T231304_20190301T233006_20230405T162425_1021_022_301_____LN3_R_NT_005.SEN3`

## Contact
For questions or collaborations, reach out to:

**Author:** [Your Name]  
**Email:** [your_email@example.com]  
**GitHub Repository:** [https://github.com/your_repo](https://github.com/your_repo)

## Acknowledgments
This project is developed as part of an academic assignment for the **GEOL0069** module at **UCL Earth Sciences Department**.
