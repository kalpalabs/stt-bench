# stt-bench
cli utility for benchmarking transcription models on Indic Datasets.
Currently supported models:
```
ai4bharat/indic-conformer-600m-multilingual, kalpalabs/Menka, gpt-4o-transcribe, deepgram-nova-3
```

Currently supported datasets:
```
IndicVoices, Lahaja, Svarah, Fleurs
```

## Usage:
1. Environment variables:
Following environment variables need to be set based on the model on which inference is to be performed:
```
HF_TOKEN, OPENAI_API_KEY, MENKA_BASE_URL, DEEPGRAM_API_KEY, SARVAM_API_KEY, GEMINI_API_KEY
```

2. Run inference of a model on multiple datasets - 
```
stt-bench run --model gpt-4o-transcribe
```
This command dumps model inference results into inference/{model}/{dataset} directory for each dataset that the inference is run on. Results are stored in csv named `*predictions.csv`. By default the code will run inference on all supported datasets. To run inference on only selected datasets, use it as:
```
stt-bench run --model gpt-4o-transcribe --eval-datasets Fleurs
```

3. Evaluate WER and CER metrics from the results directory:
```
stt-bench evaluate --dir inference/{model}
```
This will create a `evaluation_metrics.csv` within metrics/{model}/{dataset} that contains wer, cer metrics over all splits of the particular dataset, and the final row contains metrics over the entire dataset.

## Requirements
In addition to pyproject.toml, some datasets (like Lahaja and Svarah) also need ffmpeg backend to process audios. Install ffmpeg >= 6.