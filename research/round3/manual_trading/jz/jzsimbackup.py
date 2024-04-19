import pandas as pd
import numpy as np

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
        2, 3, 4, 2, 3
    ]
}

def run_simulation():
    df = pd.DataFrame(data)
    base_treasure = 7500
    num_players = 10000
    num_iterations = 100  # Including the initial and 99 nxt* iterations before final

    def calculate_probabilities(df, player_counts=None):
        df['Total Treasure'] = df['Multiplier'] * base_treasure
        if player_counts is not None:
            df['Player Percentage'] = player_counts / num_players * 100
            df['Effective Hunters'] = df['Hunters'] + df['Player Percentage']
        else:
            df['Effective Hunters'] = df['Hunters'] + 2

        df['Value Per Player'] = df['Total Treasure'] / df['Effective Hunters']
        total_value = df['Value Per Player'].sum()
        df['Probabilities'] = df['Value Per Player'] / total_value
        return df['Probabilities'].values

    # Initial simulation
    player_counts = pd.Series(0, index=df.index)
    probabilities = calculate_probabilities(df)

    for _ in range(num_iterations - 1):  # -1 because we'll do the final separately
        choices = np.random.choice(df.index, size=num_players, p=probabilities)
        player_counts = pd.Series(choices).value_counts()
        probabilities = calculate_probabilities(df, player_counts)

    # Final simulation to set the final player counts
    final_choices = np.random.choice(df.index, size=num_players, p=probabilities)
    final_player_counts = pd.Series(final_choices).value_counts()

    # Calculate final values
    df['Final Player Count'] = final_player_counts.fillna(0)
    df['Final Player Percentage'] = df['Final Player Count'] / num_players * 100
    df['Final Effective Hunters'] = df['Hunters'] + df['Final Player Percentage']
    df['Final Value Per Player'] = df['Total Treasure'] / df['Final Effective Hunters']

    return df[['Hunters', 'Multiplier', 'Final Player Count', 'Final Value Per Player']]

# Run the simulation
final_results = run_simulation()

print("Final Spot Details and Player Values:")
print(final_results)
