import numpy as np

# Sample data
data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Mean
mean = np.mean(data)
print(f"Mean: {mean}")

# Lower Quartile (Q1)
lower_quartile = np.percentile(data, 25)
print(f"Lower Quartile (Q1): {lower_quartile}")

# Upper Quartile (Q3)
upper_quartile = np.percentile(data, 75)
print(f"Upper Quartile (Q3): {upper_quartile}")

""