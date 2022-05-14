import numpy as np
import matplotlib.pyplot as plt



def plot_predicted_signals(signals: np.ndarray, title: str, figsize=(18, 10)):
    rows = int(np.ceil(np.sqrt(len(signals))))
    cols = int(np.ceil(len(signals) / rows))
    
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=figsize, constrained_layout=True, squeeze=False)
    plt.suptitle(title)
    for ax, signal in zip(axes.flat, signals):
        ax.plot(signal)
        
    plt.show()
    
    
def get_plotted_signals(predicted_classes, signals, class_: int):
    possible_classes = {0 : 'one peak', 1 : 'two peaks', 2 : 'three peaks'}
    current_plotted_class = possible_classes[class_]
    ind: np.ndarray = np.argwhere(predicted_classes == current_plotted_class).squeeze()
    plotted_signals = signals[ind].squeeze()
    
    if len(plotted_signals.shape) < 2:
        plotted_signals = plotted_signals.unsqueeze(dim=0)
        
    return current_plotted_class, plotted_signals