# Import the required libraries
from matplotlib import pyplot as plt  # For plotting data
import random                         # For generating random data

# Function to plot data as either a scatter plot or a line plot
def plot(x_values, y_values, is_scatter):
    if is_scatter == True:
        # Create a scatter plot for raw or observed data
        plt.scatter(x_values, y_values, color="orange")
    else:
        # Create a line plot for predicted/trend data
        plt.plot(x_values, y_values, color="orange")
    
    # Label the plot
    plt.title("Linear Regression Test Data")
    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    
    # Display the plot
    plt.show()

# Function to compute the mean and standard deviation of a list of values
def mean_and_stddev(values):
    mean = 0
    std_dev = 0

    # Compute mean
    for i in range(len(values)):
        mean += values[i]
    mean = mean / len(values)

    # Compute standard deviation
    for i in range(len(values)):
        std_dev += (values[i] - mean) ** 2
    std_dev = (std_dev / len(values)) ** 0.5

    return mean, std_dev

# Function to apply Z-Score normalization to x and y data
def Z_Score_Normalization(x_values, mean_x, std_dev_x, y_values, mean_y, std_dev_y):
    # Create empty arrays to store normalized values
    x_standardized = [0] * len(x_values)
    y_standardized = [0] * len(y_values)

    # Apply Z-score normalization to each value
    for i in range(len(x_values)):
        x_standardized[i] = (x_values[i] - mean_x) / std_dev_x
        y_standardized[i] = (y_values[i] - mean_y) / std_dev_y
    #Return the normalized values
    return x_standardized, y_standardized

# Main function: orchestrates data generation, normalization, and plotting
def main():
    x_values = []
    y_values = []

    # Generate 20 random (x, y) data points
    for i in range(20):
        x_values.append(random.randint(1, 500))      # Random x between 1 and 500
        y_values.append(random.randint(1, 5000))     # Random y between 1 and 5000

    """
    Polynomial Regression Setup:
    - We define w as a list of weights (slopes) for a 15-degree polynomial.
    - The model is: y = w[0] * x^15 + w[1] * x^14 + ... + w[14] * x + b
    - Initially, all weights and bias b are set to 0.
    """
    w = [0] * 15
    b = 0

    # Plot the raw (unstandardized) data
    plot(x_values, y_values, True)

    # Compute mean and standard deviation of x and y
    mean_x, std_dev_x = mean_and_stddev(x_values)
    mean_y, std_dev_y = mean_and_stddev(y_values)

    # Print the mean and standard deviation for reference
    print(f"Mean X: {mean_x}, StdDev X: {std_dev_x}")
    print(f"Mean Y: {mean_y}, StdDev Y: {std_dev_y}")

    # Normalize the x and y values using Z-score normalization
    x_standardized, y_standardized = Z_Score_Normalization(
        x_values, mean_x, std_dev_x, y_values, mean_y, std_dev_y
    )

    # Plot the normalized data to see how it's centered and scaled
    plot(x_standardized, y_standardized, True)

# Run the main function only if this script is being executed directly
if __name__ == "__main__":
    main()