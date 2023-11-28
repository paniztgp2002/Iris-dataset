# Import necessary libraries
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def genomic_classification():
    # Step 2: Load the Iris dataset
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # Step 3: Preprocess the dataset (Standardization)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Step 4: Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 5: Implement a machine learning model (Random Forest)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Step 6: Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Display the results
    print(f"Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

def main():
    # Execute the genomic classification
    genomic_classification()

if __name__ == "__main__":
    main()
