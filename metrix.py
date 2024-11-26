import os
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def calculate_average_waveform():
    waveforms = []
    for file_number in range(150):
        filename = f"audio/audio_data/{file_number}.wav"
        data, _ = librosa.load(filename)
        waveforms.append(data)
    max_length = min(len(w) for w in waveforms)
    average_waveform = np.mean(np.vstack([w[:max_length] for w in waveforms]), axis=0)
    return average_waveform

# # Example usage:
# wav_filenames = ["audio/audio_data/0.wav", "audio/audio_data/1.wav", "audio/audio_data/2.wav"]
# average_data = calculate_average_waveform()
#
# # Plot the average waveform
# plt.title("Average Waveform")
# librosa.display.waveplot(average_data)
# plt.show()
def save_waveform(audio_dir="audio/audio_data", image_dir="audio_waveform_images"):
    """Saves waveforms of WAV files as images.

    Args:
        audio_dir (str, optional): Directory containing the WAV files. Defaults to "audio/audio_data".
        image_dir (str, optional): Directory to save the generated images. Defaults to "imgs".
    """

    # Create the image directory if it doesn't exist
    os.makedirs(image_dir, exist_ok=True)

    figures = []  # List to store plot figures for batch saving
    for file_number in range(150):
        filename = os.path.join(audio_dir, f"{file_number}.wav")
        sample_rate, data = librosa.load(filename)

        fig, ax = plt.subplots()  # Create a new figure and axis for each plot
        ax.set_title(f"{file_number}")
        librosa.display.waveplot(data, sr=sample_rate, ax=ax)
        # librosa.display.waveplot(data, sr=None, ax=ax)  # Use ax to plot on specific axis

        figures.append(fig)  # Add the figure to the list

    # Save all plots in a batch
    for i, fig in enumerate(figures):
        fig.savefig(os.path.join(image_dir, f"{i}.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)  # Close the figure after saving the image


save_waveform()