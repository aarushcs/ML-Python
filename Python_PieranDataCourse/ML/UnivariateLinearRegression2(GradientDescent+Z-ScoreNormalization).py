"""
This Python script performs **simple linear regression** using **gradient descent** on a set of 
randomly generated data points. The overall purpose is to demonstrate how a linear relationship 
between input x and output y can be learned and evaluated step-by-step. 

### Key Components of the Script:

1. **Random Data Generation**: Creates 150 (x, y) pairs, where y follows a roughly linear pattern 
   with a randomized slope and intercept, simulating noisy real-world data.
   
2. **Data Visualization**: 
   - Plots the original unprocessed data.
   - Plots the data after normalization.

3. **Data Normalization**: 
   - Applies **Z-score normalization** to x and y values to improve the effectiveness 
     and stability of the gradient descent algorithm.

4. **Gradient Descent Training**:
   - Learns optimal parameters (w for slope, b for intercept) using iterative updates.
   - Reports loss every 100000 iterations to track learning progress.

5. **Unstandardization**:
   - Converts the learned slope and intercept back to the original data scale for interpretation.

6. **Model Evaluation**:
   - Uses **R-squared** (coefficient of determination) to quantify how well the model fits the data.

7. **Final Plot**:
   - Shows the regression line over the original data points to visualize the result.
"""

# Import the required libraries
from matplotlib import pyplot as plt  # For plotting data
import random                         # For generating random data

# Function to plot data as either a scatter plot or with a fitted line
def plot(x_values, y_values, w, b, message, is_scatter):
    if is_scatter == True:
        # Only scatter plot the data
        plt.scatter(x_values, y_values, color="orange")
    else:
        # Scatter plot with fitted regression line
        plt.scatter(x_values, y_values, color="orange")
        plt.plot(x_values, [w * x + b for x in x_values], color="green")  # Plot best-fit line
    plt.title(message)
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    plt.show()

# Function to compute mean and standard deviation of a list of values
def mean_and_stddev(values):
    mean = 0
    std_dev = 0

    # Calculate mean
    for i in range(len(values)):
        mean += values[i]
    mean = mean / len(values)

    # Calculate standard deviation
    for i in range(len(values)):
        std_dev += (values[i] - mean) ** 2
    std_dev = (std_dev / len(values)) ** 0.5

    return mean, std_dev

# Function to apply Z-score normalization to a list of values
def Z_Score_Normalization(x_values, mean_x, std_dev_x):
    # Normalize each value using Z-score formula
    x_standardized = [0] * len(x_values)
    for i in range(len(x_values)):
        x_standardized[i] = (x_values[i] - mean_x) / std_dev_x
    return x_standardized

# Gradient Descent to learn best-fit parameters w and b
def Gradient_Descent(w, b, x_standardized, y_standardized, learning_rate=1e-4, epochs=500000):
    dw = 0
    db = 0
    for i in range(epochs):
        for j in range(len(x_standardized)):
            # Compute partial derivatives for current sample
            dw +=  (w * x_standardized[j] + b - y_standardized[j]) * x_standardized[j]
            db += (w * x_standardized[j] + b - y_standardized[j])
        dw /= len(x_standardized)
        db /= len(x_standardized)
        w = w - learning_rate * dw
        b = b - learning_rate * db
        # Log loss periodically
        if (i % 100000 == 0 or i == epochs - 1):
            print(LossFuncCalc(w, b, x_standardized, y_standardized))
    return w, b

# Function to compute Mean Squared Error loss
def LossFuncCalc(w, b, x_standardized, y_standardized):
    loss = 0
    for i in range(len(x_standardized)):
        loss += (w * x_standardized[i] + b - y_standardized[i]) ** 2
    loss /= 2 * len(x_standardized)
    return loss

# Convert learned parameters back to original scale
def Unstandardize(w, b, std_dev_x, std_dev_y, mean_x, mean_y):
    w_unstandardized = w * (std_dev_y / std_dev_x)
    b_unstandardized = (b - w * mean_x / std_dev_x) * std_dev_y + mean_y
    return w_unstandardized, b_unstandardized

# Evaluate model using R-squared (coefficient of determination)
def R_squared(x_values, y_values, w, b, mean_y):
    R_squared_num = 0
    R_squared_den = 0
    for i in range(0, len(y_values)):
        # Sum of squared residuals
        R_squared_num += (y_values[i] - (w * x_values[i] + b)) ** 2
        # Total sum of squares
        R_squared_den += (y_values[i] - mean_y) ** 2
    R_squared = 1 - (R_squared_num / R_squared_den)
    print("R^2: ", R_squared)

# Main function: generates data, trains model, evaluates, and plots
def main():
    x_values = []
    y_values = []

    # Generate 15 random data points with noise
    for i in range(150):
        x_values.append(random.randint(-500, 500))  # Random x between -500 and 500
        # y is a random linear function of x with some noise
        y_values.append(x_values[i] * random.randint(1, 5) + random.randint(-200, 200))

    # Initialize weight and bias
    w = 0
    b = 0

    # Plot original raw data
    plot(x_values, y_values, None, None, "Original Plot", True)

    # Compute mean and standard deviation for normalization
    mean_x, std_dev_x = mean_and_stddev(x_values)
    mean_y, std_dev_y = mean_and_stddev(y_values)

    # Display computed mean and stddev
    print(f"Mean X: {mean_x}, StdDev X: {std_dev_x}")
    print(f"Mean Y: {mean_y}, StdDev Y: {std_dev_y}")

    # Normalize the data
    x_standardized = Z_Score_Normalization(x_values, mean_x, std_dev_x)
    y_standardized = Z_Score_Normalization(y_values, mean_y, std_dev_y)

    # Plot normalized data
    plot(x_standardized, y_standardized, None, None, "Plot after Z-Score Normalization", True)

    # Perform gradient descent to learn model parameters
    w, b = Gradient_Descent(w, b, x_standardized, y_standardized)

    # Convert learned parameters back to original scale
    w, b = Unstandardize(w, b, std_dev_x, std_dev_y, mean_x, mean_y)

    # Evaluate model using R-squared
    R_squared(x_values, y_values, w, b, mean_y)

    # Plot final regression line on original data
    plot(x_values, y_values, w, b, "Final Plot", False)

    #Print Equation used for Linear Regression
    print(f'Equation: y = {w}x + {b}')

# Execute main only when script is run directly
if __name__ == "__main__":
    main()