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

# # Using list comprehension to filter out values where "Hunters" are either 2 or 3
# filtered_multiplier = [m for m, h in zip(data["Multiplier"], data["Hunters"]) if h != 2 and h != 3]
# filtered_hunters = [h for h in data["Hunters"] if h != 2 and h != 3]

# # Updating the dictionary with the filtered data
# data["Multiplier"] = filtered_multiplier
# data["Hunters"] = filtered_hunters

# # Printing the updated dictionary
# print(data)

def run_simulation():
    df = pd.DataFrame(data)
    base_treasure = 7500
    num_players = 1000
    num_iterations = 1000  # Total number of iterations

    # Initialize cumulative player counts
    cumulative_player_counts = pd.Series(0, index=df.index)

    def calculate_probabilities(df, player_counts=None):
        df['Total Treasure'] = df['Multiplier'] * base_treasure
        if player_counts is not None:
            df['Player Percentage'] = player_counts / num_players * 100
            df['Effective Hunters'] = df['Hunters'] + df['Player Percentage']
        else:
            df['Effective Hunters'] = df['Hunters'] + 4

        df['Value Per Player'] = df['Total Treasure'] / df['Effective Hunters']
        total_value = df['Value Per Player'].sum()
        # df['Probabilities'] = df['Value Per Player'] / total_value
        df['Probabilities'] = (df['Value Per Player'] / total_value) ** 1  # adding bias to probabilities
        df['Probabilities'] /= df['Probabilities'].sum()  # Normalizing again
        return df['Probabilities'].values

    def get_weights(df, average_counts):
        probs = pd.DataFrame()
        probs['Weight']= calculate_probabilities(df, current_average_counts)
        df['Rank'] = probs['Weight'].rank(method='first', ascending=False)
        first_order = df[df['Rank'] <= df['Rank'].quantile(0.33)].index
        second_order = df[(df['Rank'] > df['Rank'].quantile(0.33)) & (df['Rank'] <= df['Rank'].quantile(0.67))].index
        third_order = df[(df['Rank'] > df['Rank'].quantile(0.67)) & (df['Rank'] <= df['Rank'].quantile(1))].index

        choices_random = np.random.choice(df.index, size=int(num_players * 0.2), replace=True)
        choices_first_order = np.random.choice(first_order, size=int(num_players * 0.4), replace=True)
        choices_second_order = np.random.choice(second_order, size=int(num_players * 0.3), replace=True)
        choices_third_order = np.random.choice(third_order, size=int(num_players * 0.1), replace=True)

        # Combine all choices
        choices = np.concatenate([choices_random, choices_first_order, choices_second_order, choices_third_order])
        return choices
    for i in range(num_iterations):
        current_average_counts = cumulative_player_counts / (i + 1)
        choices=get_weights(df, current_average_counts)
        # probabilities = calculate_probabilities(df, current_average_counts)
        # choices = np.random.choice(df.index, size=num_players, p=probabilities)
        player_counts = pd.Series(choices).value_counts()
        player_counts = player_counts.reindex(df.index, fill_value=0)
        cumulative_player_counts += player_counts

    # Compute final averages from cumulative counts
    final_average_player_counts = cumulative_player_counts / num_iterations
    df['Average Player Count'] = final_average_player_counts
    df['Average Player Percentage'] = df['Average Player Count'] / num_players * 100
    df['Average Effective Hunters'] = df['Hunters'] + df['Average Player Percentage']
    df['Total Treasure'] = df['Multiplier'] * base_treasure
    df['Average Value Per Player'] = df['Total Treasure'] / df['Average Effective Hunters']

    # Final iteration based on average player counts

    final_choices = get_weights(df, final_average_player_counts)
    # final_probabilities = calculate_probabilities(df, final_average_player_counts)
    # final_choices = np.random.choice(df.index, size=num_players, p=final_probabilities)
    final_player_counts = pd.Series(final_choices).value_counts()
    final_player_counts = final_player_counts.reindex(df.index, fill_value=0)

    df['Final Player Count'] = final_player_counts
    df['Final Player Percentage'] = df['Final Player Count'] / num_players * 100
    df['Final Effective Hunters'] = df['Hunters'] + df['Final Player Percentage']
    df['Final Value Per Player'] = df['Total Treasure'] / df['Final Effective Hunters']

    # Sort results by Average Value Per Player
    sorted_df = df.sort_values('Final Value Per Player', ascending=False)

    return sorted_df[['Hunters', 'Multiplier', 'Average Player Percentage', 'Average Value Per Player', 'Final Player Percentage', 'Final Value Per Player']]

# Run the simulation
final_results = run_simulation()

print("Detailed Spot Results Including Average Values and Final Iteration Results (Sorted by Average Value):")
print(final_results)
