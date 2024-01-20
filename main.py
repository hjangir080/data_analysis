import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def load_data(file_path):
    return pd.read_csv(file_path, delimiter=';')


def explore_data(dataframe):
    print("Column Names:")
    print(dataframe.columns)

    print("\nDataset Information:")
    print(dataframe.info())

    print("\nSummary Statistics:")
    print(dataframe.describe())

    print("\nFirst Few Rows of the Dataset:")
    print(dataframe.head())


def visualize_data(dataframe):
    sns.set(style="whitegrid")

    categorical_columns = ['Marital status', 'Application mode', 'Previous qualification', 'Gender']
    for column in categorical_columns:
        plt.figure(figsize=(10, 6))
        sns.countplot(x=column, hue='Target', data=dataframe, palette='Set2')
        plt.title(f'Dropout Rate by {column}')
        plt.show()

    # Numerical variables
    numerical_columns = ['Age at enrollment', 'Admission grade', 'Unemployment rate', 'Inflation rate', 'GDP']

    for column in numerical_columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(x=column, hue='Target', data=dataframe, kde=True, palette='Set2', multiple="stack")
        plt.title(f'Distribution of {column} by Dropout Status')
        plt.show()

    # Histogram for 'Age at enrollment'
    plt.figure(figsize=(10, 6))
    sns.histplot(dataframe['Age at enrollment'], bins=20, kde=True, color='skyblue')
    plt.title('Distribution of Age at Enrollment')
    plt.xlabel('Age at Enrollment')
    plt.show()

    # Scatter Plot for 'Admission grade' vs 'Curricular units 1st sem (grade)'
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Admission grade', y='Curricular units 1st sem (grade)', data=dataframe, hue='Target',
                    palette='Set2')
    plt.title('Scatter Plot: Admission Grade vs Curricular Units 1st Sem Grade')
    plt.xlabel('Admission Grade')
    plt.ylabel('Curricular Units 1st Sem Grade')
    plt.legend(title='Target')
    plt.show()

    # Pie Chart for 'Target' column
    plt.figure(figsize=(8, 6))
    dataframe['Target'].value_counts().plot.pie(autopct='%1.1f%%', colors=['lightcoral', 'lightgreen'])
    plt.title('Distribution of Target Variable')
    plt.show()

    # Pie Chart for Mother's qualification
    plt.figure(figsize=(8, 6))
    dataframe["Mother's qualification"].value_counts().plot.pie(colors=sns.color_palette('pastel'))
    plt.title("Distribution of Students' Mother's qualification")
    plt.show()

    # Pie Chart for Father's qualification
    plt.figure(figsize=(8, 6))
    dataframe["Father's qualification"].value_counts().plot.pie(colors=sns.color_palette('pastel'))
    plt.title("Distribution of Students' Father's qualification")
    plt.show()

    # Average Grades by Parents' qualification
    plt.figure(figsize=(12, 8))
    order = dataframe.groupby("Mother's qualification")['Curricular units 1st sem (grade)'].mean().sort_values().index
    sns.barplot(x="Mother's qualification", y='Curricular units 1st sem (grade)', data=dataframe, ci=None,
                palette='viridis', order=order)
    plt.title("Average Grades by Mother's qualification")
    plt.xlabel("Mother's qualification")
    plt.ylabel('Average Curricular Units 1st Sem Grade')
    plt.xticks(rotation=45)
    plt.show()

    plt.figure(figsize=(12, 8))
    order = dataframe.groupby("Father's qualification")['Curricular units 1st sem (grade)'].mean().sort_values().index
    sns.barplot(x="Father's qualification", y='Curricular units 1st sem (grade)', data=dataframe, ci=None,
                palette='viridis', order=order)
    plt.title("Average Grades by Father's qualification")
    plt.xlabel("Father's qualification")
    plt.ylabel('Average Curricular Units 1st Sem Grade')
    plt.xticks(rotation=45)
    plt.show()


def linear_regression(dataframe):
    linear_regression_data = dataframe[['Admission grade', 'Curricular units 1st sem (grade)']].dropna()
    X_train, X_test, y_train, y_test = train_test_split(
        linear_regression_data[['Admission grade']],
        linear_regression_data['Curricular units 1st sem (grade)'],
        test_size=0.2,
        random_state=42
    )
    linear_reg_model = LinearRegression()
    linear_reg_model.fit(X_train, y_train)
    y_pred = linear_reg_model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Admission grade', y='Curricular units 1st sem (grade)', data=linear_regression_data)
    plt.plot(X_test, y_pred, color='red', linewidth=2)
    plt.title('Linear Regression: Admission Grade vs Curricular Units 1st Sem Grade')
    plt.xlabel('Admission Grade')
    plt.ylabel('Curricular Units 1st Sem Grade')
    plt.show()


def kmeans_clustering(dataframe):
    clustering_data = dataframe[['Admission grade', 'Curricular units 1st sem (grade)', 'Age at enrollment',
                                 'Unemployment rate', 'Inflation rate', 'GDP']].dropna()
    scaler = StandardScaler()
    clustering_data_standardized = scaler.fit_transform(clustering_data)
    kmeans = KMeans(n_clusters=3, random_state=42)
    dataframe['Cluster'] = kmeans.fit_predict(clustering_data_standardized)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Admission grade', y='Curricular units 1st sem (grade)', hue='Cluster', data=dataframe,
                    palette='viridis',
                    legend='full')
    plt.title('KMeans Clustering: Admission Grade vs Curricular Units 1st Sem Grade')
    plt.xlabel('Admission Grade')
    plt.ylabel('Curricular Units 1st Sem Grade')
    plt.legend(title='Cluster')
    plt.show()

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def logistic_regression(dataframe):
    # Convert the 'Target' column to numeric using LabelEncoder
    le = LabelEncoder()
    dataframe['Target'] = le.fit_transform(dataframe['Target'])

    # Select features and target variable
    features = ['Admission grade', 'Curricular units 1st sem (grade)', 'Age at enrollment',
                'Unemployment rate', 'Inflation rate', 'GDP']
    target = 'Target'

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(dataframe[features], dataframe[target], test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train_standardized = scaler.fit_transform(X_train)
    X_test_standardized = scaler.transform(X_test)

    # Initialize and fit the logistic regression model
    logreg_model = LogisticRegression(random_state=42)
    logreg_model.fit(X_train_standardized, y_train)

    # Predictions on the test set
    y_pred = logreg_model.predict(X_test_standardized)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # Confusion matrix and classification report
    confusion_mat = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print("\nConfusion Matrix:")
    print(confusion_mat)

    print("\nClassification Report:")
    print(class_report)


from sklearn.preprocessing import LabelEncoder


def find_correlation(dataframe):
    le = LabelEncoder()
    dataframe['Target'] = le.fit_transform(dataframe['Target'])
    correlation = dataframe.corr()
    target_correlation = correlation['Target']
    print(target_correlation.sort_values(ascending=False))
    # Convert the correlation results to a DataFrame
    correlation_df = target_correlation.to_frame()

    # Create a heatmap
    plt.figure(figsize=(10, 5))
    sns.heatmap(correlation_df, annot=True, cmap='coolwarm')
    plt.title('Correlation with Target Variable')
    plt.show()



file_path = '/Users/rakeshjangir/Desktop/ida miniproject/data.csv'
dataframe = load_data(file_path)

explore_data(dataframe)
find_correlation(dataframe)
logistic_regression(dataframe)
visualize_data(dataframe)
linear_regression(dataframe)
kmeans_clustering(dataframe)

