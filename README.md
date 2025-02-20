# Sea Ice and Lead Unsupervised Classification
   - Use K-Means and GMM to identify distinct clusters in the dataset.
   - Compare cluster assignments to ESA ground truth.
**Evaluation**:
   - Generate classification reports.
   - Compute confusion matrices to measure accuracy.

### Sample Code for GMM:
```python
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import numpy as np

# Sample data
X = np.random.rand(100, 2)

# GMM model
gmm = GaussianMixture(n_components=3)
gmm.fit(X)
y_gmm = gmm.predict(X)

# Plotting
plt.scatter(X[:, 0], X[:, 1], c=y_gmm, cmap='viridis')
centers = gmm.means_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.title("Gaussian Mixture Model")
plt.show()
```

---

## Results
### Clustering Outcomes

After applying clustering, we extract the waveform clusters and visualize the results.

**Example Waveform Classification:**

![image](https://github.com/user-attachments/assets/68d705b6-facb-4b82-b51e-5510ecd0ba29)


The left plot represents sea ice echoes, while the right plot corresponds to lead echoes. The peak intensity and spread provide insight into different surface properties.

### Clustered Data Scatter Plot

Scatter plots of clustered data using key Sentinel-3 features:

![Clustered Data](./images/scatter_plot.png)

#### Comparison with ESA Data
To validate the clustering results, we compute a confusion matrix:
```python
from sklearn.metrics import confusion_matrix, classification_report

true_labels = flag_cleaned - 1  # Adjust ESA labels
predicted_gmm = clusters_gmm  # Predicted GMM clusters

conf_matrix = confusion_matrix(true_labels, predicted_gmm)
print("Confusion Matrix:")
print(conf_matrix)

class_report = classification_report(true_labels, predicted_gmm)
print("Classification Report:")
print(class_report)
```

---

## Installation
To run this project, install the required dependencies:
```bash
pip install rasterio
pip install netCDF4
```

---

## Data
The dataset is extracted from Sentinel-3 satellite data using the Copernicus Data Space.

Example files:
- **Sentinel-2 Optical Data**: `S2A_MSIL1C_20190301T235611_N0207_R116_T01WCU_20190302T014622.SAFE`
- **Sentinel-3 OLCI Data**: `S3B_SR_2_LAN_SL_20190301T231304_20190301T233006_20230405T162425_1021_022_301______LN3_R_NT_005.SEN3`

---

## Contact
For questions or contributions, contact:
- **Project Author**: Affan Mazlan
- **Email**: zcfbabi@ucl.ac.uk
- **Project Repository**: [GitHub Repository](https://github.com/affan1317/sea-ice-and-lead-unsupervised-learning)

---

## Acknowledgments
- This project is part of an assignment for module **GEOLO069** taught at **UCL Earth Sciences Department**.
- ESA and Copernicus Data Space provided Sentinel-3 altimetry data.
