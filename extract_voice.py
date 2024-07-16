import argparse
import torch
torch.set_num_threads(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", default='test.mp3',
                        help="Path of the audio file of which the voice is to be extracted.", type=str)
    parser.add_argument("--out_path", default = 'speech.wav',
                        help="Path of the output file", type=str)
    parser.add_argument("--save_speech", default=True,
                        help="Flag to save the audio with no speech, only other sounds", type=bool)
    args = parser.parse_args()

    file_path = args.file_path
    out_path = args.out_path
    save_speech = args.save_speech

    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=False)

    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

    sampling_rate = 16000

    wav = read_audio(file_path, sampling_rate=sampling_rate)

    # get speech timestamps from full audio file
    speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=sampling_rate)
    if save_speech:
        no_speech_timestamps = []
        start = speech_timestamps[0]['start']
        end = speech_timestamps[0]['end']
        no_speech_timestamps.append({'start': 0, 'end': start})
        for i, fragment in enumerate(speech_timestamps[1:(len(speech_timestamps))]):
            start = end
            no_speech_timestamps.append({'start': start, 'end': fragment['start']})
            end = fragment['end']
        start = speech_timestamps[-1]['end']
        end = len(wav)
        no_speech_timestamps.append({'start': start, 'end': end})
        speech_timestamps = no_speech_timestamps

    print(speech_timestamps)
    # # If timestamps are to be saved in a seperate .txt file
    # with open('timestamps.txt', 'w') as file:
    #     file.write(str(speech_timestamps))

    # Merging all speech chunks into one file
    save_audio(out_path, collect_chunks(speech_timestamps, wav), sampling_rate=sampling_rate) 

if __name__ == "__main__":
    main()