import pandas as pd
import matplotlib.pyplot as plt

# Data provided
data = {
    "Multiplier":
        [
        24,70,41,21,60,
        47,82,87,80,35,
        73,89,100,90,17,
        77,83,85,79,55,
        12,27,52,15,30
        ],
    "Hunters": 
        [
        2,4,3,2,4,
        3,5,5,5,3,
        4,5,8,7,2,
        5,5,5,5,4,
        2,3,4,2,3,
        ]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Create an array of tuples where each tuple is (Number of Hunters, Multiplier)
hunter_multiplier_pairs = list(zip(df['Hunters'], df['Multiplier']))

# Display the array of pairs
print(hunter_multiplier_pairs)
# Plotting the data
plt.figure(figsize=(10, 6))
plt.scatter(df['Hunters'], df['Multiplier'], color='blue', alpha=0.5)
plt.title('Scatter Plot of Hunters vs Multiplier')
plt.xlabel('Hunters')
plt.ylabel('Multiplier')
plt.grid(True)
plt.show()
