import matplotlib.pyplot as plt


def get_dataset_histogram(data_df, graph_title):
    # Get data for each label
    data_0 = data_df[data_df['label'] == 0]
    data_1 = data_df[data_df['label'] == 1]

    # Plot the histogram for each group
    fig, ax = plt.subplots()

    # Plot label 0 (Human) with blue color
    ax.hist(data_0['label'], bins=1, alpha=0.7, rwidth=0.8, color='#1f77b4', label=None)

    # Plot label 1 (AI) with orange color
    ax.hist(data_1['label'], bins=1, alpha=0.7, rwidth=0.8, color='#ff7f0e', label=None)

    # Manually add legend labels
    custom_legend_labels = ['Human', 'AI']
    custom_colors = ['#1f77b4', '#ff7f0e']
    custom_handles = [plt.Line2D([0], [0], color=color, lw=4) for color in custom_colors]

    # Add legend
    ax.legend(custom_handles, custom_legend_labels, title='Label', loc='upper right')

    # Set custom x-axis labels to show 'Human' and 'AI'
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Human', 'AI'])

    # Add titles and labels
    plt.title(graph_title)
    plt.xlabel('Source')
    plt.ylabel('Frequency')

    # Show the plot
    plt.show()


def get_bar_graph(model1_val, model2_val, model3_val, metrics, title):
    import matplotlib.pyplot as plt
    import numpy as np

    # Confusion matrix components for three models
    model1 = model1_val
    model2 = model2_val
    model3 = model3_val

    # Bar chart settings
    x = np.arange(len(metrics))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plotting the bars for each model
    rects1 = ax.bar(x - width, model1, width, label='Bag of Words', color='blue')
    rects2 = ax.bar(x, model2, width, label='DistilBERT', color='green')
    rects3 = ax.bar(x + width, model3, width, label='RoBERTa', color='red')

    # Adding labels, title, and custom x-axis tick labels
    ax.set_xlabel('Confusion Matrix Components')
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45, ha='right')  # Rotate labels for better readability
    ax.legend()

    # Adding data labels on top of each bar
    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., height + 1, f'{int(height)}', ha='center', va='bottom')

    # Display the plot
    plt.tight_layout()
    plt.show()
