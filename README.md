# LexTime: A Benchmark for Temporal Reasoning in Legal Texts

## Introduction

This repository contains the data and code for **LexTime**, a benchmark designed for evaluating temporal reasoning in legal texts on the task of event ordering.

## LexTime Dataset

LexTime is extracted from labor-related federal complaints in the United States.  
We provide the dataset in the [`data/`](data/) directory.

The [`data/splits`](data/splits) folder contains the fully formatted dataset in [`lextime_512samples.csv`](data/splits/lextime_512samples.csv) and its key subsets:
- [`long_context.csv`](data/splits/long_context.csv): Contains paragraphs with more than 150 tokens.
- [`short_context.csv`](data/splits/short_context.csv): Contains paragraphs with fewer than 150 tokens.
- [`pairs_explicit_implicit.csv`](data/splits/pairs_explicit_implicit.csv): Queries where one event is implicit and the other is explicit.
- [`pairs_explicit.csv`](data/splits/pairs_explicit.csv): Queries where both events are explicit.

The dataset is stored in CSV format, with each row containing a **paragraph**, a **label**, and a **query**.

The initial dataset annotations are provided in [`data/annotations/annotated_complaints.csv`](data/annotations/annotated_complaints.csv), with the following columns:
- **paragraph**: Context paragraph extracted from the text of the complaints.
- **query**: Temporal relation query, containing 2 events and a temporal relationship.
- **label**: Binary label (Yes/No).
- **error (temporal/events)**: Error type.
- **readability (1-4)**: Readability score of the query.
- **relevance (1-4)**: Relevance score.
- **event type (implicit/explicit)**: Categorization of events.
- **relevant/irrelevant context paragraph**: Classification of the context relevance to a the legal process.
- **NOTES**: Additional notes.


## Experiments 

We conducted our experiments using **Python 3.8.10**.  
To install dependencies, run: `pip install -r requirements.txt` (our requirements use CUDA 12).

## Language Analysis: Specificities of Legal Language
In the [`code/language_analysis`](code/language_analysis) directory, we provide the code to run the linguistic analysis of our dataset, showing the specificities of the legal language. We compare **LexTime** with the **TRACIE** dataset that contains short stories, and we provide the results in the [code/language_analysis/outputs](code/language_analysis/outputs) folder for both datasets. 

## Inference and Error Analysis
In the [`code/inference`](code/inference) directory, we provide the code as a shell script to run the experiments to test the capabilities of the models on our tasks.

In the paper, we test the following models with and without chain-of-thought (CoT) and obtain the following results (on the whole dataset, 512 samples). Results for specific subsets and splits are in the paper. Each score is the average of three runs.

| Model                 | ZS   | 1S   | FS   |
|-----------------------|------|------|------|
| GPT-4o               | 69.9 | 74.9 | 77.4 |
| *+CoT*               | -    | 68.6 | 72.6 |
| GPT-4 Turbo          | 70.2 | 75.1 | 77.2 |
| *+CoT*               | -    | 77.6 | 74.3 |
| Mistral$_{123B}$     | 61.9 | 70.1 | 73.9 |
| *+CoT*               | -    | 67.8 | 72.4 |
| LLaMA 3.1$_{70B}$    | 66.1 | 71.3 | 71.9 |
| *+CoT*               | -    | 69.3 | 66.6 |
| LLaMA 3.1$_{8B}$     | 51.9 | 60.5 | 55.9 |
| *+CoT*               | -    | 48.4 | 54.6 |
| LLaMA 3.1$_{8B}$ (Base) | 49.6 | 50.5 | 50.2 |
| *+CoT*               | -    | 52.3 | 50.8 |
| LLaMA 3.2$_{3B}$     | 52.2 | 53.5 | 51.0 |
| *+CoT*               | -    | 53.8 | 57.2 |
| LLaMA 3.2$_{3B}$ (Base) | 50.6 | 52.3 | 52.9 |
| *+CoT*               | -    | 52.9 | 48.8 |
| Flan-T5$_{Large}$      | 48.5 | 51.6 | 52.1 |
| *+CoT*               | -    | 50.2 | 53.7 |

In [`code/error_analysis`](code/error_analysis), we provide the random sample of 100 errors used in our paper\'s error analysis, as well as the outputs categorized by error type in each csv file.




