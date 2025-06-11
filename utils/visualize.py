import matplotlib.pyplot as plt


def plot_fft_heatmap(data, window, sampling_frequency_hz=160):
    import numpy as np
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    """
    Plot FFT magnitude heatmaps for the first window and both sensors.
    
    Args:
        data: np.ndarray of shape (5, 32, 492, 2), already np.abs(FFT)
        sampling_frequency_hz: float, the sampling frequency in Hz
    """
    assert data.shape == (5, 16, 492, 2), "Expected shape (5, 32, 492, 2)"
    
    num_freqs = data.shape[1]
    freq_axis = np.fft.fftshift(np.fft.fftfreq(num_freqs, d=1 / sampling_frequency_hz))
    
    window_data = data[window]  # shape (32, 492, 2), shifted freq bins
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    fig.suptitle(f"FFT Magnitude - Window {window}", fontsize=14)

    for sensor in range(2):
        ax = axes[sensor]
        heatmap = window_data[:, :, sensor]  # shape: (32, 492)

        im = ax.imshow(
            heatmap,
            aspect='auto',
            origin='lower',
            extent=[0, 492, freq_axis[0], freq_axis[-1]],
            interpolation='nearest',
            cmap='viridis'
        )
        ax.set_title(f"Sensor {sensor}")
        ax.set_xlabel("Range Bin")
        ax.set_ylabel("Frequency (Hz)")

    # Proper colorbar placement
    cbar = fig.colorbar(im, ax=axes[1], orientation='vertical', shrink=0.85, pad=0.02)
    cbar.set_label('FFT Magnitude')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    pass


def histogramm(arr):
    flat_arr = arr.flatten()

    plt.hist(flat_arr, bins=200)  # Adjust bins if needed
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of All Values in the Array')
    plt.show()


def plot_confusion_matrix(model, dataset):
    import numpy as np
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    y_true = dataset.test._labels
    y_pred = np.argmax(model.predict(dataset.test._data), axis=-1)

    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Display it
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45) # type: ignore
    plt.title("Confusion Matrix")
    plt.show()


def visualize_model(model):
    import visualkeras
    from keras.models import Sequential
    from keras.layers import Dropout, Activation, Dense, Conv2D
    filtered_layers = [layer for layer in model.layers if not isinstance(layer, Dropout)]
    w_activation_layers = []
    for layer in filtered_layers:
        w_activation_layers.append(layer)
        if isinstance(layer, Conv2D) or isinstance(layer, Dense):
            layer.name += ' + ReLU'
    m = Sequential(w_activation_layers)
    visualkeras.layered_view(m, draw_volume=True, legend=True).show()
