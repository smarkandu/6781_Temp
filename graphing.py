import pandas as pd
import matplotlib.pyplot as plt


def get_dataset_histogram(data_df, graph_title):
    # Get data for each label
    data_0 = data_df[data_df['label'] == 0]
    data_1 = data_df[data_df['label'] == 1]

    # Plot the histogram for each group
    fig, ax = plt.subplots()
    ax.hist(data_0['label'], bins=1, alpha=0.7, rwidth=0.8, color='#1f77b4', label='0')
    ax.hist(data_1['label'], bins=1, alpha=0.7, rwidth=0.8, color='#ff7f0e', label='1')

    # Set custom x-axis labels to show 0 and 1
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Human', 'AI'])

    # Add titles and labels
    plt.title(graph_title)
    plt.xlabel('Label')
    plt.ylabel('Frequency')

    # Show the legend
    plt.legend()

    # Show the plot
    plt.show()
