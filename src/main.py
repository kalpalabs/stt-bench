import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from jiwer import wer, cer
from tqdm import tqdm
from datasets import get_dataset_split_names
import torch
import typer
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from whisper.normalizers import EnglishTextNormalizer
from functools import lru_cache

from .models import IndicConformerModel, MenkaModel, GPT4oTranscribeModel, DeepgramNova3Model, SarvamAIModel, GeminiModel
from .tasks import STTDataset, _STT_BENCH_DATASET

torch.set_grad_enabled(False)

_MODEL_CLASSES = {
    "ai4bharat/indic-conformer-600m-multilingual": IndicConformerModel,
    "kalpalabs/Menka": MenkaModel,
    "gpt-4o-transcribe": GPT4oTranscribeModel,
    "deepgram-nova-3": DeepgramNova3Model,
    "sarvamai/saarika": SarvamAIModel,
    "google/gemini-2.5-pro": GeminiModel,
}

_DATASET_OPTIONS = ["IndicVoices", "Lahaja", "Svarah", "Fleurs", "Vaani-transcription-part"]
_INFERENCE_DIR = "inference/"
_METRICS_DIR = "metrics/"
_NORMALIZER_FACTORY = None


def get_dataset(name: str):
    if name == "IndicVoices":
        return STTDataset(
            dataset_name="IndicVoices",
            audio_column_name="audio_filepath",
            transcript_column_name="text",
            language_column_name="lang",
        )
    elif name == "Lahaja":
        return STTDataset(
            dataset_name="Lahaja",
            audio_column_name="audio_filepath",
            transcript_column_name="text",
            language_column_name="lang",
        )
    elif name == "Svarah":
        return STTDataset(dataset_name="Svarah", audio_column_name="audio_filepath", transcript_column_name="text", language_column_name="lang")
    elif name == "Fleurs":
        return STTDataset(
            dataset_name="Fleurs",
            audio_column_name="audio",
            transcript_column_name="transcription",
            language_column_name="lang_iso639",
        )
    elif name == "Vaani-transcription-part":
        return STTDataset(
            dataset_name="Vaani-transcription-part",
            audio_column_name="audio",
            transcript_column_name="transcript",
            language_column_name="lang_iso639",
        )
    else:
        raise ValueError(f"{name} not in {_DATASET_OPTIONS}")


app = typer.Typer(
    no_args_is_help=True, add_completion=True, help="STT Benchmark CLI for evaluating models and calculating metrics."
)


@lru_cache(maxsize=None)
def get_normalizer(language):
    global _NORMALIZER_FACTORY
    if _NORMALIZER_FACTORY is None:
        _NORMALIZER_FACTORY = IndicNormalizerFactory()

    if language == 'en':
        return EnglishTextNormalizer()
    elif language in {"brx", "doi", "kok", "mai"}: # Script used by these languages is Devanagari, so we use Hindi normalizer.
        return _NORMALIZER_FACTORY.get_normalizer("hi", remove_nuktas=False).normalize
    # Urdu normalization doesn't work correctly with indic-nlp-library right now
    # raised an issue here: https://github.com/anoopkunchukuttan/indic_nlp_library/issues/77
    elif language and language != "ur" and _NORMALIZER_FACTORY.is_language_supported(language):
        return _NORMALIZER_FACTORY.get_normalizer(language, remove_nuktas=False).normalize
    return None


def normalize(text, language):
    normalized = text
    # Remove content within special tags that are present for Vaani dataset.
    normalized = re.sub(r"<[^>]*>", " ", normalized)
    normalized = re.sub(r"\{[^}]*\}", " ", normalized)
    normalized = re.sub(r"\[[^\]]*\]", " ", normalized)
    normalized = re.sub(r"\([^)]*\)", " ", normalized)
    normalized = re.sub(r"--", " ", normalized)
    normalized = " ".join(normalized.split())
    
    normalizer = get_normalizer(language)
    if normalizer:
        normalized = normalizer(normalized)

    normalized = normalized.replace("\u0964", " ")
    normalized = normalized.replace("\u0965", " ")
    normalized = normalized.replace(",", " ")
    
    normalized = " ".join(normalized.split())
    return normalized

@app.command(
    help="Calculate WER and CER metrics from all CSV files within a directory containing ground truth and transcripts."
)
def evaluate(model_path: str = typer.Option(..., "--dir", help="Path to directory containing predictions csv files of a particular model")):
    for path in next(os.walk(model_path))[1]:
        path = os.path.join(model_path, path)
        csv_fnames = sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith("predictions.csv")])
        if not csv_fnames:
            print(f"No 'predictions.csv' files found in '{path}'.")
            continue
        
        results_path = path.replace(_INFERENCE_DIR, _METRICS_DIR)
        os.makedirs(results_path, exist_ok=True)
        results_path = f"{results_path}/evaluation_metrics.csv"
        results = []

        model_name = ""
        dataset_name = ""

        for csv_file in tqdm(csv_fnames, desc="Evaluating CSV files"):
            df = pd.read_csv(csv_file)
            if df["ground_truth"].isna().any() or df["transcript"].isna().any():
                print(
                    f"Error: Missing ground_truth ({df['ground_truth'].isna().sum()}) or transcript ({df['transcript'].isna().sum()}) in '{csv_file}'. Skipping {csv_file}."
                )
                continue

            # Calculate WER over list of samples instead of averaging WER over individual samples.
            df["normalized_ground_truth"] = df.apply(lambda row: normalize(row["ground_truth"], row["language"]), axis=1)
            df["normalized_transcript"] = df.apply(lambda row: normalize(row["transcript"], row["language"]), axis=1)

            model_name = df["model_name"].iloc[0]
            dataset_name = df["dataset_name"].iloc[0]

            results.append(
                {
                    "model_name": model_name,
                    "dataset_name": dataset_name,
                    "split": df["split"].iloc[0],
                    "wer": 100 * wer(df["normalized_ground_truth"].tolist(), df["normalized_transcript"].tolist()),
                    "cer": 100 * cer(df["normalized_ground_truth"].tolist(), df["normalized_transcript"].tolist()),
                }
            )

        if results:
            overall_wer = sum(r["wer"] for r in results) / len(results)
            overall_cer = sum(r["cer"] for r in results) / len(results)
            results.append(
                {
                    "model_name": model_name,
                    "dataset_name": dataset_name,
                    "split": "all",
                    "wer": overall_wer,
                    "cer": overall_cer,
                }
            )
            results_df = pd.DataFrame(results)
            results_df.to_csv(results_path, index=False)

            print(
                f"Overall WER = {overall_wer:.2f}, Overall CER = {overall_cer:.2f} for {model_name} over {dataset_name}"
            )
            print(f"Saved evaluation metrics to: {os.path.abspath(results_path)}")
        else:
            print(f"No valid results found for {path}.")
        print("\n")


def run_model_on_dataset(model, dataset, split: str, concurrency: int = 6):
    split_dataset = dataset.dataset_dict[split]
    split_language = split_dataset[0][dataset.language_column_name]
    if split_language not in model.supported_languages:
        print(f"Skipping {split} as it is not supported by {model.name}.")
        return None

    num_samples = len(split_dataset)

    if num_samples == 0:
        print(f"No samples found for split '{split}'. Skipping...")
        return None
    print(f"Evaluating {dataset.dataset_name}/{split}")

    ground_truths = []
    transcripts = []
    languages = []

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {}
        for audio, sr, ground_truth, language in dataset.__iter__(split):
            future = executor.submit(model.transcribe, audio=audio, sampling_rate=sr, language=language)
            futures[future] = (ground_truth, language)

        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                transcription = future.result()
                transcripts.append(transcription)

                ground_truths.append(futures[future][0])
                languages.append(futures[future][1])
            except Exception as e:
                print(f"Error transcribing audio: {e}. Skipping this sample.")

    df = pd.DataFrame(
        {
            "ground_truth": ground_truths,
            "transcript": transcripts,
            "dataset_name": [dataset.dataset_name] * len(ground_truths),
            "split": [split] * len(ground_truths),
            "model_name": [model.name] * len(ground_truths),
            "language": languages,
        }
    )
    return df


@app.command(help="Run a specified model on datasets and save transcripts to CSV.")
def run(
    model_name: str = typer.Option(
        ...,
        "--model",
        help="The model name to evaluate",
        show_choices=True,
        case_sensitive=False,
        show_default=False,
        # Provide list of available models from _MODEL_CLASSES
        metavar="{" + ",".join(_MODEL_CLASSES.keys()) + "}",
    ),
    eval_datasets: list[str] = typer.Option(
        None,
        "--eval-datasets",
        help="Datasets to evaluate on",
        show_choices=True,
        case_sensitive=False,
        show_default=False,
        metavar="{" + ",".join(_DATASET_OPTIONS) + "}",
    ),
    do_evaluate: bool = typer.Option(
        True,
        "--do-evaluate",
        help="Evaluate WER & CER metrics",
    ),
):
    model_cls = _MODEL_CLASSES[model_name]
    model = model_cls(model_name)

    if eval_datasets is None:
        eval_datasets = _DATASET_OPTIONS

    for dataset_name in eval_datasets:
        dataset = get_dataset(dataset_name)
        output_dir = os.path.join(_INFERENCE_DIR, model_name.split('/')[-1], dataset_name.split('/')[-1])
        os.makedirs(output_dir, exist_ok=True)

        dataset_splits = get_dataset_split_names(_STT_BENCH_DATASET, dataset_name)
        for split in dataset_splits:
            try:
                df_split = run_model_on_dataset(model, dataset, split)
            except Exception as e:
                print(f"Error evaluating model on dataset {dataset_name} split {split}: {e}")
                continue

            if df_split is not None and not df_split.empty:
                csv_path = f"{output_dir}/{split}.predictions.csv"
                df_split.to_csv(csv_path, index=False)
                print(f"Dumped transcripts for {dataset_name}/{split} to {os.path.abspath(csv_path)}")

    if do_evaluate and eval_datasets:
        evaluate(os.path.dirname(output_dir))


if __name__ == "__main__":
    app()
