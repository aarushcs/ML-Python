"""
This Python script generates a scatter plot of random linear data points and calculates the line of best fit using linear regression.
It first creates random x and y values, then computes the slope and y-intercept of the best-fit line based on the formula for linear regression.
The script also calculates the mean of both x and y values, the sum of squared residuals, the standard deviation of residuals, and the R-squared value, 
which indicates the goodness of the fit. Afterward, it displays the scatter plot of the data points and the line of best fit. 
The code concludes by printing the slope, y-intercept, standard deviation, and R-squared value to the console.
"""

from matplotlib import pyplot as plt
import random
import sys

x_values = []
y_values= []

#Creates random x and y values that vary, and put all x values in order in x_values, and all y values in array in order in y_values
for i in range(0,50):
    slope = random.uniform(1,15)
    y_int = random.uniform(0,1000)
    x_values.append(random.randint(-1000,1000))
    y_values.append(x_values[i]*slope+y_int)

# Shows the plot before proceeding with linear regression.
plt.scatter(x_values,y_values, color="orange")
plt.title("Linear Regression Test Data")
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.show()

#Calculating the mean of y-values
mean_y = 0
sum_y = 0
for i in y_values:
    sum_y=sum_y+i
mean_y=sum_y/len(y_values)

#Calculating the mean of x-values
mean_x=0
sum_x=0
for i in x_values:
    sum_x=sum_x+i
mean_x=sum_x/len(x_values)


#Calculating the line of best fit's slope using a formula used during linear regression, that I proved using differiential calculus
slope_num = 0
slope_den=0
for i in range(0,len(x_values)):
    slope_num = slope_num + (x_values[i]-mean_x)*(y_values[i]-mean_y)
    slope_den = slope_den + (x_values[i]-mean_x)**2
slope=slope_num/slope_den
print("Slope: ",slope)

#Using the mean to calculate the y-intercept
y_int=mean_y-(slope*mean_x)
print("y-intercept: ",y_int)

#Calculating the sum of squares for the residuals compared to the starting equation of y=(mean of y-values)
ss_mean=0
for i in range(0,len(y_values)):
    ss_mean=ss_mean+(y_values[i]-(slope*x_values[i]+y_int))**2
print("Sum of Squared Residuals: ", ss_mean)

#Calculating the standard deviations for the residuals compared to the starting equation of y=(mean of y-values)
stdev_mean=(ss_mean/(len(y_values)-2))**0.5
#This is to bypass Python's float precision
if abs(stdev_mean)<1e-10:
    print(0)
else:
    print("Standard Deviation: ",stdev_mean)
    plt.scatter(x_values, y_values, color="orange")

#Calculating R^2, the coeffecient of determination
R_squared_num = 0
R_squared_den = 0
for i in range(0,len(y_values)):
    R_squared_num = R_squared_num + (y_values[i]-(slope*x_values[i]+y_int))**2
    R_squared_den = R_squared_den + (y_values[i]-mean_y)**2
R_squared=1-(R_squared_num/R_squared_den)
print("R^2: ",R_squared)

#Plotting the line of best fit
plt.plot(x_values, [slope * x + y_int for x in x_values], color="green")  # The best fit line
plt.title("Linear Regression Test Data with Line of Best Fit")
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.show()