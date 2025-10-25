from datasets import load_dataset
import os

_STT_BENCH_DATASET = "kalpalabs/stt-bench"


class STTDataset:
    def __init__(self, dataset_name: str, audio_column_name: str, transcript_column_name: str, language_column_name: str = None, seed: int = 42, max_samples: int = 500):
        self.dataset_name = dataset_name
        
        self.dataset_dict = load_dataset(_STT_BENCH_DATASET, self.dataset_name, num_proc=os.cpu_count())
        for split in self.dataset_dict:
            self.dataset_dict[split] = self.dataset_dict[split].shuffle(seed=seed)
            self.dataset_dict[split] = self.dataset_dict[split].select(range(min(max_samples, len(self.dataset_dict[split]))))
        
        self.audio_column_name = audio_column_name
        self.transcript_column_name = transcript_column_name
        self.language_column_name = language_column_name

    def __iter__(self, split: str):
        for sample in self.dataset_dict[split]:
            audio_data = sample[self.audio_column_name].get_all_samples()
            audio, sr = audio_data.data, audio_data.sample_rate
            lang = sample.get(self.language_column_name) if self.language_column_name is not None else None
            
            yield audio, sr, sample[self.transcript_column_name], lang


class SvarahDataset(STTDataset):
    def __iter__(self, split: str):
        for sample in self.dataset_dict[split]:
            audio_data = sample[self.audio_column_name].get_all_samples()
            audio, sr = audio_data.data, audio_data.sample_rate
            yield audio, sr, sample[self.transcript_column_name], "en"  # Svarah dataset is in English
