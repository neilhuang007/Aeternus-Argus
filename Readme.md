# **VPN and Malicious Network Request Classifier Using AI**

## **Project Overview🚀**
This project applies machine learning and AI techniques to classify VPN traffic and identify malicious network requests. It utilizes **time-based flow features** such as `flowBytesPerSecond` and `flowPktsPerSecond` to distinguish between:
- **VPN vs. Non-VPN** traffic
- **Malicious vs. Normal** network flows

The project emphasizes data-driven insights using robust visualizations and automated file management for traffic datasets.

---

## **Key Features**

1. **Traffic Classification**:
   - Uses time-related flow-based features for high-accuracy traffic classification.
   - Two primary tasks:
     - **VPN Detection**: Distinguishes VPN from non-VPN flows.
     - **Malicious Detection**: Identifies malicious network requests based on file types.

2. **Advanced Visualizations**:
   - Generates the following for deeper traffic insights:
     - **eCDF Plots**: Empirical Cumulative Distribution of flow features.
     - **Heatmaps**: Correlation between flow features for VPN and non-VPN.
     - **Scatter Plots**: Relationship between `flowBytesPerSecond` and `flowPktsPerSecond`.

3. **Automated Dataset Management**:
   - Downloads, extracts, and organizes VPN datasets from ISCX repositories.
   - Processes test cases and classifies files into `normal` and `malicious` folders.

4. **Concurrent Processing**:
   - Speeds up file classification using **ThreadPoolExecutor**.

---

## **Project Structure**

```plaintext
├── DataVisualizer.py       # Generates eCDF, Heatmaps, and Scatter Plots
├── RequestClassifier.py    # Classifies network files into 'normal' and 'malicious'
├── SetupData.py            # Downloads and unpacks VPN datasets
├── Readme.md               # Project documentation
├── graphs/                 # Output directory for visualizations
│   ├── TimeBasedFeatures-Dataset-15s-VPN_ecdf.png
│   ├── TimeBasedFeatures-Dataset-15s-VPN_heatmap.png
│   ├── TimeBasedFeatures-Dataset-15s-VPN_scatter.png
├── vpndata/                # Downloaded VPN datasets
├── testcases/              # Input network requests
├── normal/                 # Classified normal files
├── malicious/              # Classified malicious files
└── requirements.txt        # Python dependencies
```

---

## **Dataset Sources**
1. **VPN Traffic Dataset**:
   - Retrieved from ISCX VPN-NonVPN Traffic Dataset.
   - [Download Link](http://205.174.165.80/CICDataset/ISCX-VPN-NonVPN-2016/)

2. **Malicious Network Traffic**:
   - Extracted from the open-source repository [BlazeHTTP](https://github.com/chaitin/blazehttp).

---

## **Installation and Setup**

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/YourRepository/VPN-Traffic-Classifier.git
   cd VPN-Traffic-Classifier
   ```

2. **Install Dependencies**:
   Ensure Python 3.8+ is installed. Then run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download VPN Dataset**:
   Run the setup script to download and extract datasets automatically:
   ```bash
   python SetupData.py
   ```

4. **Generate Visualizations**:
   Run the visualizer script to generate eCDF plots, heatmaps, and scatter plots:
   ```bash
   python DataVisualizer.py
   ```

5. **Classify Requests**:
   Classify files into normal and malicious categories:
   ```bash
   python RequestClassifier.py
   ```

---

## **Visualizations**

### **1. eCDF Plot**  
The eCDF plot compares the empirical cumulative distribution of `flowBytesPerSecond` for VPN and Non-VPN traffic:

![eCDF Plot](graphs/TimeBasedFeatures-Dataset-15s-VPN_ecdf.png)

**Observations**:
- The majority of data flows are concentrated at lower byte rates for both VPN and non-VPN users.
- Some outliers suggest significantly higher flow rates for non-VPN traffic.

---

### **2. Heatmaps**  
Heatmaps show the correlation between `flowBytesPerSecond` and `flowPktsPerSecond` for both VPN and non-VPN traffic:

![Heatmaps](graphs/TimeBasedFeatures-Dataset-15s-VPN_heatmap.png)

**Observations**:
- VPN users exhibit a stronger correlation (~0.8) between the two features.
- Non-VPN traffic has a lower correlation (~0.31), suggesting more variability in flow behavior.

---

### **3. Scatter Plots**  
Scatter plots depict the relationship between `flowBytesPerSecond` and `flowPktsPerSecond` for VPN and non-VPN traffic:

![Scatter Plots](graphs/TimeBasedFeatures-Dataset-15s-VPN_scatter.png)

**Observations**:
- VPN traffic clusters tightly at lower values.
- Non-VPN traffic shows more scattered outliers with significantly higher flow rates.

---

## **Machine Learning Models**
The project utilizes the following models for classification:
1. **C4.5 Decision Tree**:
   - High accuracy and explainability with tree-structured outputs.
   - Outperforms KNN slightly in precision and recall.

2. **K-Nearest Neighbors (KNN)**:
   - Simple and effective for flow-based classification.
   - Useful for comparative analysis.

---

## **Key Findings**

1. **Time-based Flow Features**:
   - Features like `flowBytesPerSecond` and `flowPktsPerSecond` are effective for distinguishing VPN and non-VPN traffic.

2. **Shorter Flow Timeouts Improve Accuracy**:
   - Classifiers perform better with shorter flow durations (15s timeout) as opposed to longer durations (120s).

3. **Visual Insights**:
   - **Heatmaps** reveal stronger feature correlations in VPN traffic.
   - **Scatter Plots** highlight differences in flow behavior between VPN and non-VPN users.

---

## **Future Work**
- Integrate additional classifiers such as **Random Forest** and **SVM**.
- Develop real-time traffic analysis for live network data.
- Explore more flow features to improve malicious traffic detection.

---

## **References**
1. Draper-Gil, Gerard, et al. *"Characterization of Encrypted and VPN Traffic using Time-related Features"*. (ICISSP 2016)
2. ISCX VPN-NonVPN Traffic Dataset: [Dataset Link](http://205.174.165.80/CICDataset/ISCX-VPN-NonVPN-2016/)
3. BlazeHTTP Repository: [GitHub Link](https://github.com/chaitin/blazehttp)

---

## **Contact**
- **Author**: Neil Huang
- **Email**: neilhuang007@gmail.com

---
