import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score

# Need to install numpy scikit-learn first using commond: pip install numpy scikit-learn


def main():
    try:
        # Read N
        N = int(input("Enter the number of points N (positive integer): "))
        if N <= 0:
            raise ValueError("N must be a positive integer")

        # Read k
        k = int(input("Enter the number of neighbors k (positive integer): "))
        if k <= 0:
            raise ValueError("k must be a positive integer")

        # Check if k <= N
        if k > N:
            raise ValueError("k must be less or equal to N")

        # Initialize an array to store the points
        points = np.zeros((N, 2))

        # Read N points
        for i in range(N):
            x = float(input(f"Enter x value for point {i+1}: "))
            y = float(input(f"Enter y value for point {i+1}: "))
            points[i] = [x, y]

        # Split points into X and Y
        X_train = points[:, 0].reshape(-1, 1)  # Features
        y_train = points[:, 1]  # Target

        # Create and train the model
        model = KNeighborsRegressor(n_neighbors=k)
        model.fit(X_train, y_train)

        # Read X for prediction
        X_test = float(input("Enter the value of X for prediction: "))
        X_test = np.array([[X_test]])

        # Predict Y
        y_pred = model.predict(X_test)
        print(f"The predicted value of Y is: {y_pred[0]}")

        # Calculate the coefficient of determination (R^2)
        y_train_pred = model.predict(X_train)
        r2 = r2_score(y_train, y_train_pred)
        print(f"Coefficient of determination (R^2): {r2}")

    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
