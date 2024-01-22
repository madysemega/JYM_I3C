import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def plot_dataset(df, values, title, apply_scaling=False):
    plt.figure(figsize=(10, 6))  # Adjust the size of the plot as needed

    # Define a list of colors and markers
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']
    markers = ['o', '^', 's', 'D', '*', 'x']

    # Create a copy of the dataframe to apply scaling if needed
    df_scaled = df.copy()
    
    if apply_scaling:
        scaler = MinMaxScaler()
        df_scaled[values] = scaler.fit_transform(df[values])

    for i, value in enumerate(values):
        # Use modulo operator to cycle through colors and markers if there are more values than colors/markers
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]

        if apply_scaling:
            plt.plot(df_scaled.index, df_scaled[value], color=color, marker=marker, linestyle='-', linewidth=2, label=value)
        else:
            plt.plot(df.index, df[value], color=color, marker=marker, linestyle='-', linewidth=2, label=value)

    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Scaled Value' if apply_scaling else 'Value')
    plt.grid(True)
    plt.legend()  # Add a legend to distinguish the different lines
    plt.show()


plt.style.use('ggplot')
def plot_predictions(df_train, df_test, df_baseline):
    plt.figure(figsize=(30, 15))  # Adjust the size of the plot as needed

    # Plotting the training data
    plt.plot(df_train.index, df_train.value, color="gray", linestyle='-', linewidth=2, label='Train Data')
    
    # Plotting the actual test values
    plt.plot(df_test.index, df_test.value, color="black", linestyle='-', linewidth=2, label='Actual Test Values')

    # Plotting the baseline predictions
    # plt.plot(df_baseline.index, df_baseline.prediction, color="blue", linestyle='--', linewidth=2, label='Linear Regression')

    # Plotting the model predictions
    plt.plot(df_test.index, df_test.prediction, color="red", linestyle='--', linewidth=2, label='Model Predictions')

    # Adding lines to indicate the distance between actual and predicted values
    for idx in df_test.index:
        plt.plot([idx, idx], [df_test.loc[idx, 'value'], df_test.loc[idx, 'prediction']], color='purple', linestyle=':', linewidth=1)

    # Adding title and labels
    plt.title("Model Predictions vs Actual Values")
    plt.xlabel("Time")
    plt.ylabel("Value")

    # Adding a legend
    plt.legend()

    # Show the plot
    plt.show()
    
