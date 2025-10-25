# uv run --project=stt-bench stt-bench --help
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from jiwer import wer, cer
from tqdm import tqdm
from datasets import get_dataset_split_names
import torch
import typer

from .models import IndicConformerModel, MenkaModel, GPT4oTranscribeModel, DeepgramNova3Model
from .tasks import STTDataset, SvarahDataset, _STT_BENCH_DATASET

torch.set_grad_enabled(False)

_MODEL_CLASSES = {
    "ai4bharat/indic-conformer-600m-multilingual": IndicConformerModel,
    "kalpalabs/Menka": MenkaModel,
    "gpt-4o-transcribe": GPT4oTranscribeModel,
    "deepgram-nova-3": DeepgramNova3Model,
}

_DATASET_CLASSES = {
    "IndicVoices": STTDataset(
        dataset_name="IndicVoices",
        audio_column_name="audio_filepath",
        transcript_column_name="text",
        language_column_name="lang",
    ),
    "Lahaja": STTDataset(
        dataset_name="Lahaja",
        audio_column_name="audio_filepath",
        transcript_column_name="text",
        language_column_name="lang",
    ),
    "Svarah": SvarahDataset(dataset_name="Svarah", audio_column_name="audio_filepath", transcript_column_name="text"),
    "fleurs": STTDataset(
        dataset_name="fleurs",
        audio_column_name="audio",
        transcript_column_name="transcription",
        language_column_name="lang_iso639",
    ),
}

app = typer.Typer(
    no_args_is_help=True, add_completion=True, help="STT Benchmark CLI for evaluating models and calculating metrics."
)


@app.command(help="Calculate WER and CER metrics from a CSV file containing ground truth and transcripts.")
def evaluate(path: str = typer.Argument(..., help="Path to the CSV file")):
    df = pd.read_csv(path)
    num_na = df["transcript"].isna().sum()
    print(f"Number of NaN transcripts: {num_na} out of {len(df)} samples")

    df["transcript"] = df["transcript"].fillna("")
    df["wer"] = df.apply(
        lambda r: wer(r["ground_truth"], r["transcript"]),
        axis=1,
    )
    df["cer"] = df.apply(
        lambda r: cer(r["ground_truth"], r["transcript"]),
        axis=1,
    )

    for dataset_name, dataset_df in df.groupby("split"):
        dataset_wer = dataset_df["wer"].mean()
        dataset_cer = dataset_df["cer"].mean()
        print(
            f"WER = {dataset_wer * 100:.2f}, CER = {dataset_cer * 100:.2f} for split '{dataset_name}' over {len(dataset_df)} samples"
        )
    print("\n")

    avg_wer = df["wer"].mean()
    avg_cer = df["cer"].mean()
    print(f"Average WER = {avg_wer * 100:.2f}, Average CER = {avg_cer * 100:.2f} over the entire dataset\n")


def run_model_on_dataset(model, dataset, split: str, concurrency: int = 8):
    split_dataset = dataset.dataset_dict[split]
    num_samples = len(split_dataset)

    if num_samples == 0:
        print(f"No samples found for split '{split}'. Skipping...")
        return pd.DataFrame(columns=["ground_truth", "transcript", "dataset_name", "split"])
    print(f"Evaluating {dataset.dataset_name}/{split}")

    ground_truths = []
    transcripts = []

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {}
        for audio, sr, ground_truth, language in dataset.__iter__(split):
            future = executor.submit(model.transcribe, audio=audio, sampling_rate=sr, language=language)
            futures[future] = ground_truth

        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                transcription = future.result()
                transcripts.append(transcription)
                ground_truths.append(futures[future])
            except Exception as e:
                print(f"Error transcribing audio: {e}. Skipping this sample.")

    df_split = pd.DataFrame(
        {
            "ground_truth": ground_truths,
            "transcript": transcripts,
            "dataset_name": [dataset.dataset_name] * len(ground_truths),
            "split": [split] * len(ground_truths),
        }
    )
    return df_split


@app.command(help="Run a specified model on datasets and save transcripts to CSV.")
def run(model_name: str = typer.Argument(..., help="The model name to evaluate (e.g., kalpalabs/Menka)")):
    model_cls = _MODEL_CLASSES[model_name]
    model = model_cls()

    for dataset_name, dataset in _DATASET_CLASSES.items():
        df = pd.DataFrame(columns=["ground_truth", "transcript", "dataset_name", "split"])
        dataset_splits = get_dataset_split_names(_STT_BENCH_DATASET, dataset_name)
        try:
            for split in dataset_splits:
                try:
                    df_split = run_model_on_dataset(model, dataset, split)
                    df = pd.concat([df, df_split], ignore_index=True)
                except Exception as e:
                    print(f"Error evaluating model on dataset {dataset_name} split {split}: {e}")
        except KeyboardInterrupt:
            output_path = f"{model_name.split('/')[-1]}_{dataset_name}.csv"
            df.to_csv(output_path, index=False)
            print(f"\nCtrl+C detected. Partial transcripts for {dataset_name} saved to {os.path.abspath(output_path)}")
            return

        output_path = f"{model_name.split('/')[-1]}_{dataset_name}.csv"
        df.to_csv(output_path, index=False)
        print(f"Dumped transcripts for {dataset_name} to {os.path.abspath(output_path)}")


if __name__ == "__main__":
    app()
