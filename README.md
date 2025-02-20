# Sea Ice and Lead Unsupervised Classification
   - Use K-Means and GMM to identify distinct clusters in the dataset.
   - Compare cluster assignments to ESA ground truth.
   - Generate classification reports.
   - Compute confusion matrices to measure accuracy.

### Sample Code for GMM:
```python
# Python code for K-means clustering
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

# Sample data
X = np.random.rand(100, 2)

# K-means model
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

# Plotting
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.show()
```
![image](https://github.com/user-attachments/assets/e3aff6ef-910f-48d5-b4c8-ac62a2d330ad)

---

## Results
### Clustering Outcomes

After applying clustering, we extract the waveform clusters and visualize the results.

**Example Waveform Classification:**

![image](https://github.com/user-attachments/assets/92f8c9cf-5deb-4dde-b35a-0c08fecde820)
![image](https://github.com/user-attachments/assets/50068d3a-00e7-4f2a-9f37-7cccd6db9c37)



The left plot represents sea ice echoes, while the right plot corresponds to lead echoes. The peak intensity and spread provide insight into different surface properties.

### Clustered Data Scatter Plot

Scatter plots of clustered data using key Sentinel-3 features:

![image](https://github.com/user-attachments/assets/8456718c-7385-4992-aab2-86e771befc85)
![image](https://github.com/user-attachments/assets/b76806df-31b0-4c38-981b-955ed45528de)
![image](https://github.com/user-attachments/assets/5a93179c-280d-4554-abda-0be06b79fa8e)

#### Comparison with ESA Data
To validate the clustering results, we compute a confusion matrix:
```python
from sklearn.metrics import confusion_matrix, classification_report

true_labels = flag_cleaned_modified   # true labels from the ESA dataset
predicted_gmm = clusters_gmm          # predicted labels from GMM method

# Compute confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_gmm)

# Print confusion matrix
print("Confusion Matrix:")
print(conf_matrix)

# Compute classification report
class_report = classification_report(true_labels, predicted_gmm)

# Print classification report
print("\nClassification Report:")
print(class_report)
```

Classification Report:
              precision    recall  f1-score   support

         0.0       1.00      1.00      1.00      8878
         1.0       0.99      0.99      0.99      3317

    accuracy                           1.00     12195
   macro avg       1.00      1.00      1.00     12195

weighted avg       1.00      1.00      1.00     12195

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
- **Project Author**: Yiheng Shi
- **Email**: zcfbshi@ucl.ac.uk
- **Project Repository**: [GitHub Repository](https://github.com/affan1317/sea-ice-and-lead-unsupervised-learning)

---

## Acknowledgments
- This project is part of an assignment for module **GEOLO069** taught at **UCL Earth Sciences Department**.
- ESA and Copernicus Data Space provided Sentinel-3 altimetry data.
