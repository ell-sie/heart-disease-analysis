# Heart Disease Analysis

This repository contains an analysis of a heart disease dataset using various unsupervised learning techniques to cluster patients based on their medical history and identify risk factors associated with heart disease.

## Dataset

The dataset used in this analysis is the Heart Disease dataset from the UCI Machine Learning Repository. It contains 303 instances with 14 attributes including age, sex, chest pain type, blood pressure, serum cholesterol, and a target variable indicating the presence or absence of heart disease.

## Analysis Steps

1. **Load the dataset and perform EDA**: The dataset is loaded into a pandas DataFrame and exploratory data analysis (EDA) is performed to understand the distribution and characteristics of the data.
2. **Preprocess the dataset**: The dataset is preprocessed by handling missing values, scaling numerical features, and encoding categorical variables to prepare it for clustering algorithms.
3. **Apply clustering algorithms (K-means, Hierarchical, DBSCAN)**: Various clustering algorithms such as K-means, hierarchical clustering, and DBSCAN are applied to group patients based on their medical history.
4. **Visualize the clusters using PCA and t-SNE**: Principal Component Analysis (PCA) and t-Distributed Stochastic Neighbor Embedding (t-SNE) are used to visualize the clusters and gain insights into the relationships between variables.
5. **Apply Gaussian Mixture Models (GMMs)**: Gaussian Mixture Models (GMMs) are used to identify risk factors associated with heart disease by modeling the data as a mixture of multiple Gaussian distributions.
6. **Evaluate clustering performance**: The performance of the clustering algorithms is evaluated using metrics such as the Silhouette Score and Davies-Bouldin Index.
7. **Compare the performance of different clustering algorithms**: The performance of different clustering algorithms is compared to choose the one that gives the best results.

## Results

The analysis results include clustering performance metrics such as the Silhouette Score and Davies-Bouldin Index for each of the clustering algorithms used. Visualizations generated using PCA and t-SNE provide insights into the clusters formed by each algorithm.

## Repository Structure

- `heart_disease_analysis.ipynb`: Jupyter notebook containing the code, analysis, and visualizations.
- `README.md`: Description of the analysis and results.

## Usage

To replicate this analysis, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/heart-disease-analysis.git
    ```
2. Navigate to the repository directory:
    ```bash
    cd heart-disease-analysis
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Run the Jupyter notebook:
    ```bash
    jupyter notebook heart_disease_analysis.ipynb
    ```

## Requirements

- Python 3.x
- Jupyter Notebook
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Scipy

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.

## Acknowledgments

- UCI Machine Learning Repository for providing the Heart Disease dataset.
