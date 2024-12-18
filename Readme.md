# **VPN and Malicious Network Request Classifier Using AI**

## **Project Overview 🚀**
This project employs advanced machine learning techniques to classify VPN traffic and detect malicious network requests. It uses **time-based flow features** and comprehensive visualizations to analyze and distinguish between:
- **VPN vs. Non-VPN** traffic
- **Malicious vs. Normal** network flows

Key updates include feature-pair evaluation and accuracy tracking for model optimization, utilizing datasets from ISCX repositories.

---

## **Key Features**

1. **Traffic Classification**:
   - Leverages flow-based features for precise traffic categorization.
   - Focus areas:
     - **VPN Detection**: Identifies VPN vs. non-VPN flows.
     - **Malicious Traffic Detection**: Categorizes files into normal or malicious requests.

2. **Feature-Pair Analysis**:
   - Automated evaluation of feature pairs for model accuracy using `XGBoost`.
   - **Results**: Displays best feature pairs and their respective accuracy for different datasets.

3. **Dynamic Visualizations**:
   - Generates plots for deeper understanding:
     - **eCDF**: Distribution analysis of flow features.
     - **Heatmaps**: Feature correlations in VPN vs. Non-VPN traffic.
     - **Scatter Plots**: Relationships between `flowBytesPerSecond` and `flowPktsPerSecond`.
     - **Fold-wise Accuracy Charts**: Visualizes classifier performance across K-folds.

4. **Concurrent File Processing**:
   - Implements **ThreadPoolExecutor** to boost classification speed.

5. **GPU-Enhanced Model Training**:
   - Leverages XGBoost's `hist` method for faster computation during training.

---

## **Project Structure**

```plaintext
├── Train.py                # Main script for model training and feature pair evaluation
├── DataVisualizer.py       # Generates eCDF, heatmaps, and scatter plots
├── RequestClassifier.py    # Classifies files into 'normal' and 'malicious'
├── SetupData.py            # Downloads and organizes VPN datasets
├── Readme.md               # Project documentation
├── graphs/                 # Output directory for plots and visualizations
├── vpndata/                # VPN dataset files
├── testcases/              # Input files for classification
├── normal/                 # Classified normal files
├── malicious/              # Classified malicious files
└── requirements.txt        # Python dependencies
```

---

## **New Results**

### Dataset: **TimeBasedFeatures-Dataset-15s-VPN**
- **Best Feature Pair**: `('min_flowiat', 'flowPktsPerSecond')`
- **Accuracy**: **0.7824**

### Dataset: **TimeBasedFeatures-Dataset-120s-VPN**
- **Best Feature Pair**: `('min_biat', 'min_flowiat')`
- **Accuracy**: **0.7644**

---

## **Installation and Usage**

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/YourRepository/VPN-Traffic-Classifier.git
   cd VPN-Traffic-Classifier
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download and Organize Datasets**:
   ```bash
   python SetupData.py
   ```

4. **Train Model and Evaluate Features**:
   ```bash
   python Train.py
   ```

5. **Generate Visualizations**:
   ```bash
   python DataVisualizer.py
   ```

6. **Classify Requests**:
   ```bash
   python RequestClassifier.py
   ```

---

## **Visual Insights**

### **1. Accuracy per Fold (Best Pair)**
Displays K-fold accuracy for the best feature pair:

![Accuracy Plot](graphs/Accuracy-Plot-BestPair.png)

### **2. eCDF Plot**
Analyzes flow rate distribution for VPN and Non-VPN traffic:

![eCDF Plot](graphs/TimeBasedFeatures-Dataset-15s-VPN_ecdf.png)

---

## **Future Enhancements**
- Incorporate classifiers like **Random Forest** and **SVM**.
- Develop a real-time traffic classification module.
- Extend feature exploration for improved detection capabilities.

---

## **References**
1. Draper-Gil, Gerard, et al. *"Characterization of Encrypted and VPN Traffic using Time-related Features"* (ICISSP 2016).
2. ISCX VPN-NonVPN Traffic Dataset: [Download Link](http://205.174.165.80/CICDataset/ISCX-VPN-NonVPN-2016/)
3. BlazeHTTP Repository: [GitHub Link](https://github.com/chaitin/blazehttp)

---

## **Contact**
- **Author**: Neil Huang
- **Email**: neilhuang007@gmail.com

---