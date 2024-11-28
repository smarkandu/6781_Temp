import pandas as pd
import matplotlib.pyplot as plt


def get_dataset_histogram(data_df, graph_title):
    # Get data for each label
    data_0 = data_df[data_df['label'] == 0]
    data_1 = data_df[data_df['label'] == 1]

    # Plot the histogram for each group
    fig, ax = plt.subplots()

    # Plot label 0 (Human) with blue color
    ax.hist(data_0['label'], bins=1, alpha=0.7, rwidth=0.8, color='#1f77b4', label='Human')

    # Plot label 1 (AI) with orange color
    ax.hist(data_1['label'], bins=1, alpha=0.7, rwidth=0.8, color='#ff7f0e', label='AI')

    # Set custom x-axis labels to show 'Human' and 'AI'
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Human', 'AI'])

    # Add titles and labels
    plt.title(graph_title)
    plt.xlabel('Source')
    plt.ylabel('Frequency')

    # Explicitly set the legend labels
    plt.legend(title='Label', loc='upper right')

    # Show the plot
    plt.show()

