import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt


def plot_errorbar(x: list, y: dict, title: str, xlabel: str, ylabel: str, save_path: str = None):
    y_mean = [np.mean(y[i]) for i in x]
    array_plus = [np.max(y[i]) - np.mean(y[i]) for i in x]
    array_minus = [np.mean(y[i]) - np.min(y[i]) for i in x]
    fig = go.Figure(data=go.Scatter(
        x=x,
        y=y_mean,
        error_y=dict(
            type='data',
            symmetric=False,
            array=array_plus,
            arrayminus=array_minus)
    ))
    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="#7f7f7f"
        )
    )
    return fig


def plot_errorbar_plt(x: list, y: dict, title: str, xlabel: str, ylabel: str, save_path: str = None):
    y_mean = [np.mean(y[i]) for i in x]
    array_plus = [np.max(y[i]) - np.mean(y[i]) for i in x]
    array_minus = [np.mean(y[i]) - np.min(y[i]) for i in x]
    y_err = [array_minus, array_plus]

    # Create a Matplotlib figure with error bars
    plt.errorbar(x, y_mean, yerr=y_err, fmt='o', capsize=5, color='blue')  # , label='Data')

    # Connect the dots with lines
    plt.plot(x, y_mean, linestyle='-', marker='o', color='blue')  # , label='Line')

    # Set plot title and axis labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Set font properties
    plt.rcParams['font.size'] = 18

    # Show legend
    # plt.legend()


if __name__ == '__main__':
    # y = {x[0]: [], x[1]: [], x[2]: [],
    #      x[3]: [], x[4]: [], x[5]: []}

    data = "WR"
    if data == "WR":
        x = [0.02, 0.05, 0.1, 0.2, 1, 2]
        x = [f"{str(i)}%" for i in x]

        y = {x[0]: [0.51, 0.5215, 0.5193], x[1]: [0.5768, 0.5929, 0.4847], x[2]: [0.4627, 0.514, 0.6182],
             x[3]: [0.482, 0.4425, 0.55], x[4]: [0.4534, 0.4311, 0.9989], x[5]: [0.5174, 0.9763, 0.8725]}
    elif data == "size":
        x = [12, 48, 96, 180, 360]
        x = [f"{str(i)}%" for i in x]

        y = {x[0]: [0.9457, 0.7694, 0.958], x[1]: [0.9825, 1, 1], x[2]: [0.9918, 1, 1],
             x[3]: [0.9981, 1, 1], x[4]: [0.9876, 1, 1]}

    fig = plot_errorbar_plt(x, y, "AUC vs WR", "WR", "AUC")
    plt.show()
