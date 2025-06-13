# Import the required libraries
from matplotlib import pyplot as plt  # For plotting data
import random                         # For generating random data

# Function to plot data as either a scatter plot or a line plot
def plot(x_values, y_values, w, b, message, is_scatter):
    if is_scatter == True:
        # Create a scatter plot for raw or observed data
        plt.scatter(x_values, y_values, color="orange")
    else:
        plt.scatter(x_values, y_values, color="orange")
        plt.plot(x_values, [w * x + b for x in x_values], color="green")  # The best fit line
    plt.title(message)
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
def Z_Score_Normalization(x_values, mean_x, std_dev_x):
    # Create empty arrays to store normalized values
    x_standardized = [0] * len(x_values)
    # Apply Z-score normalization to each value
    for i in range(len(x_values)):
        x_standardized[i] = (x_values[i] - mean_x) / std_dev_x
    #Return the normalized values
    return x_standardized

#Gradient Descent Function
def Gradient_Descent(w,b,x_standardized,y_standardized, learning_rate=1e-3,epochs=1000):
    dw=0
    db=0
    for i in range(epochs):
        for j in range(len(x_standardized)):
            dw =  (w * x_standardized[j] + b - y_standardized[j]) * x_standardized[j]
            db = (w * x_standardized[j] + b - y_standardized[j])
            w = w - learning_rate * dw
            b = b - learning_rate * db
        dw /= len(x_standardized)
        db /= len(x_standardized)
        if (i % 100 == 0 or i == epochs-1):
            print(LossFuncCalc(w,b,x_standardized,y_standardized))
    return w,b

def LossFuncCalc(w,b,x_standardized,y_standardized):
    loss = 0
    for i in range(len(x_standardized)):
        loss += (w * x_standardized[i] + b - y_standardized[i]) ** 2
    loss /= 2 * len(x_standardized)
    return loss



def R_squared(x_values, y_values,w,b, mean_y):
    R_squared_num = 0
    R_squared_den = 0
    for i in range(0,len(y_values)):
        R_squared_num = R_squared_num + (y_values[i]-(w*x_values[i]+b))**2
        R_squared_den = R_squared_den + (y_values[i]-mean_y)**2
    R_squared=1-(R_squared_num/R_squared_den)
    print("R^2: ",R_squared)


# Main function: orchestrates data generation, normalization, and plotting
def main():
    x_values = []
    y_values = []

    # Generate 15 random (x, y) data points
    for i in range(15):
        x_values.append(random.randint(1, 500))      # Random x between 1 and 500
        y_values.append(x_values[i]*random.randint(1,5)+random.randint(-200,200))     # Random y such that slope is between 1 and 5 and y-int is between 200 and -200


    w = 0
    b = 0

    # Plot the raw (unstandardized) data
    plot(x_values, y_values,None, None,"Original Plot", True)

    # Compute mean and standard deviation of x and y
    mean_x, std_dev_x = mean_and_stddev(x_values)
    mean_y, std_dev_y = mean_and_stddev(y_values)

    # Print the mean and standard deviation for reference
    print(f"Mean X: {mean_x}, StdDev X: {std_dev_x}")
    print(f"Mean Y: {mean_y}, StdDev Y: {std_dev_y}")

    # Normalize the x and y values using Z-score normalization
    x_standardized = Z_Score_Normalization(
        x_values, mean_x, std_dev_x
    )

    y_standardized = Z_Score_Normalization(
        y_values, mean_y, std_dev_y
    )

    # Plot the normalized data to see how it's centered and scaled
    plot(x_standardized, y_standardized, None, None,"Plot after Z-Score Normalization", True)

    w,b = Gradient_Descent(w,b,x_standardized,y_standardized)
    
    R_squared(x_values, y_values,w,b, mean_y)
    plot(x_values,y_values,w,b,"Final Plot",False)

# Run the main function only if this script is being executed directly
if __name__ == "__main__":
    main()