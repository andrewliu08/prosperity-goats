import pandas as pd

# Data provided
data = {
    "Multiplier": [
        24, 70, 41, 21, 60,
        47, 82, 87, 80, 35,
        73, 89, 100, 90, 17,
        77, 83, 85, 79, 55,
        12, 27, 52, 15, 30
    ],
    "Hunters": [
        2, 4, 3, 2, 4,
        3, 5, 5, 5, 3,
        4, 5, 8, 7, 2,
        5, 5, 5, 5, 4,
        2, 3, 4, 2, 3,
    ]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Adding a new column for the ratio of Multipliers to Hunters adjusted by adding 4 to Hunters
df['Ratio'] = df['Multiplier'] / (df['Hunters'] + 4)

# Sorting the DataFrame by the 'Ratio' column
sorted_df = df.sort_values(by='Ratio', ascending=False)

# Printing the sorted DataFrame
print(sorted_df)
