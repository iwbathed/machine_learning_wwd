import os
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#### LOADING THE VOICE DATA FOR VISUALIZATION ###
walley_sample = "audio/audio_data/62.wav"
data, sample_rate = librosa.load(walley_sample)

##### VISUALIZING WAVE FORM ##
plt.title("Wave Form")
librosa.display.waveplot(data, sr=sample_rate)
plt.show()

##### VISUALIZING MFCC #######
mfccs = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40)
print("Shape of mfcc:", mfccs.shape)

plt.title("MFCC")
librosa.display.specshow(mfccs, sr=sample_rate, x_axis='time')
plt.show()

##### Doing this for every sample ##
#
# all_data = []
#
# data_path_dict = {
#     0: ["audio/background_sound/" + file_path for file_path in os.listdir("audio/background_sound/")],
#     1: ["audio/audio_data/" + file_path for file_path in os.listdir("audio/audio_data/")]
# }
#
# # the background_sound/ directory has all sounds which DOES NOT CONTAIN wake word
# # the audio_data/ directory has all sound WHICH HAS Wake word
# print(1)
# for class_label, list_of_files in data_path_dict.items():
#     for single_file in list_of_files:
#         audio, sample_rate = librosa.load(single_file) ## Loading file
#         mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40) ## Apllying mfcc
#         mfcc_processed = np.mean(mfcc.T, axis=0) ## some pre-processing
#         all_data.append([mfcc_processed, class_label])
#     print(f"Info: Succesfully Preprocessed Class Label {class_label}")
#
# df = pd.DataFrame(all_data, columns=["feature", "class_label"])
#
# ###### SAVING FOR FUTURE USE ###
# pickle_path = "audio/final_audio_data_csv"
# if not os.path.exists(pickle_path):
#     os.mkdir(pickle_path)
# df.to_pickle(pickle_path+"/audio_data2.csv")









def calculate_average_waveform():
    waveforms = []
    for file_number in range(150):
        filename = f"audio/audio_data/{file_number}.wav"
        data, _ = librosa.load(filename)
        waveforms.append(data)
    max_length = min(len(w) for w in waveforms)
    average_waveform = np.mean(np.vstack([w[:max_length] for w in waveforms]), axis=0)
    return average_waveform
# wav_filenames = ["audio/audio_data/0.wav", "audio/audio_data/1.wav", "audio/audio_data/2.wav"]
# average_data = calculate_average_waveform()
#
# # Plot the average waveform
# plt.title("Average Waveform")
# librosa.display.waveplot(average_data)
# plt.show()

def save_waveform(audio_path="audio/audio_data/", path_to_save_images="bg_waveform_images/"):
    for file_number in range(10):
        filename = f"{audio_path}{file_number}.wav"
        data, _ = librosa.load(filename)
        librosa.display.waveplot(data)
        plt.title(f"{file_number}")
        plt.savefig(f"{path_to_save_images}{file_number}.png", dpi=300, bbox_inches="tight")
        plt.close()

# save_waveform("audio/background_sound/", "bg_waveform_images/")

# filename = f"{audio_path}{file_number}.wav"
data, _ = librosa.load(r"D:\programing\my_projects\python\speech_recognition\wake_word_detection\audio\background_sound\1-all-quiet-on-the-western-front-2022-web-dlrip-avc-ukrger-sub-ukreng-hurtom_part_89.mp3")
librosa.display.waveplot(data)
# plt.title(f"{file_number}")
plt.savefig(f"bg_waveform_images/q.png", dpi=300, bbox_inches="tight")
plt.close()