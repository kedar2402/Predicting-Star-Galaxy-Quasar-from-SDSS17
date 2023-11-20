# Predicting-Star-Galaxy-Quasar-from-SDSS17
Dataset= https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17/

The provided Python code is a machine learning script that utilizes the scikit-learn library to perform classification on a stellar dataset. Here is a breakdown of the code:

1. **Importing Libraries:**
   - `numpy` and `pandas` are used for data manipulation and analysis.
   - `matplotlib.pyplot` is employed for creating visualizations.
   - `sklearn` is used for machine learning tasks, specifically for support vector machines (SVM) and k-nearest neighbors (KNN) classification.
   - `MinMaxScaler` is applied for feature scaling.
   - `classification_report` and `confusion_matrix` are used to evaluate model performance.
   - `train_test_split` is utilized to split the dataset into training and testing sets.

2. **Loading Data:**
   - The script loads a stellar classification dataset from a CSV file using `pd.read_csv`.

3. **Data Exploration:**
   - Descriptive statistics and the first few rows of the dataset are displayed using `describe()` and `head()`.

4. **Data Preprocessing:**
   - The target variable 'class' is separated from the features (X and y).

5. **K-Nearest Neighbors (KNN) Classification:**
   - A KNN classifier with three neighbors is instantiated and trained on the dataset.
   - The model is evaluated on a test set, and a classification report is generated and displayed.

6. **Support Vector Machine (SVM) Classification:**
   - Three SVM classifiers with different kernel functions ('linear', 'poly', 'rbf') are instantiated.
   - The dataset is scaled using `MinMaxScaler`.
   - Each SVM model is trained, evaluated on a test set, and a classification report is displayed.

7. **Conclusion:**
   - The script demonstrates the application of both KNN and SVM algorithms for stellar classification, providing insights into the performance of the models through classification reports. Additionally, feature scaling is applied to enhance the SVM models' effectiveness. Consideration should be given to further refining the models and exploring tuning parameters for improved accuracy.
