# hmeg -- speaking and translation exercises generator

[![Unit-tests](https://github.com/yurytsoy/hmeg/actions/workflows/tests.yml/badge.svg)](https://github.com/yurytsoy/hmeg/actions/workflows/tests.yml)

Help me, Erik Gunnemark -- a library for generating exercises to practice basic speaking constructs.

## Table of Contents

- [Installation](#installation)
  - [Via pip](#via-pip)
  - [Via git](#via-git)
- [Usage](#usage)
  - [Python](#python)
  - [Command line](#command-line)
  - [Generating exercises using Ollama](#generating-exercises-using-ollama)
  - [Configuration file](#configuration-file)
- [Format of exercises and vocabulary](#format-of-exercises-and-vocabulary)
- [Why I made this library](#why-i-made-this-library)

The idea is that mastering these building blocks helps with faster speaking and constructing more complex sentences.

Exercises are generated randomly, so they can sometimes be grammatically or semantically odd.
As long as a sentence is not abusive and is grammatically correct, it is considered a valid exercise.
The goal is to facilitate quickfire translation into Korean, where the element of surprise can aid memorization.

# Installation

## Via pip

Install the latest stable version from PyPI:

```bash
pip install hmeg
```

## Via git

Clone the repository and install in development mode:

```bash
git clone https://github.com/yurytsoy/hmeg.git
cd hmeg
pip install -e .
```

# Usage

## Python

### Default engine ("templates")

```python
from hmeg import usecases, ExerciseGenerator, load_minilex

num_exercises = 10  # number of randomly generated exercises for the selected topic

usecases.register_grammar_topics()
vocab = load_minilex()  # load words from the Minilex.

exercises = ExerciseGenerator.generate_exercises(
    topic_name="While / -(으)면서", num=num_exercises, vocab=vocab
)
print("\n".join(exercises))
```

### "Ollama" engine

```python
from hmeg import entities, usecases, ExerciseGenerator, load_minilex

num_exercises = 10  # number of randomly generated exercises for the selected topic
ollama_model = "gemma3:4b"  # Ollama model to use. Model must be pulled in advance.

if not usecases.is_ollama_available(ollama_model):  # as a precaution, not mandatory.
    exit(0)

usecases.register_grammar_topics()
exercises = ExerciseGenerator.generate_exercises(
    topic_name="While / -(으)면서",
    num=num_exercises,
    vocab_level="C1",
    engine=entities.ExerciseGenerationEngine.OLLAMA,
    model=ollama_model,
)
print("\n".join(exercises))
```

## Command line

The CLI tool `hmeg` is available after installation of the wheel.

Update file [hmeg.conf](hmeg.conf) to select the grammatical topic and number of exercises,
then run:
```bash
hmeg
```

You can also specify command-line arguments to define configuration file, topic, and/or number of generated exercises.

* Run with a custom configuration file (use the `run` subcommand):
```bash
hmeg run --config="custom/configuration/file.toml"
```

* Run with a custom topic and number of exercises:
```bash
hmeg run -n 15 -t "Have, Don’t have, There is, There isn’t / 있어요, 없어요"
```

* You can provide a partial topic name. All topics that contain the specified string will be used:
```bash
hmeg run -n 15 -t "있어요, 없어요"
hmeg run -n 15 -t "there is"
```

* List available grammar topics:
```bash
hmeg list
```

* Print help:

```bash
hmeg --help
hmeg run --help
hmeg list --help
```

## Generating exercises using Ollama

You can use [Ollama](https://ollama.com/) to generate exercises. Follow the official install instructions for your platform.

Recommended models:
* `gemma3` -- `4b` and `12b` work pretty well comparing to other families that I tried.
* `qwen3` -- `4b-instruct` is also not bad and much (much) faster than the thinking variant.

Note on `exaone3.5` (2026.01.24): I had high hopes, since the models were prepared by LG. Tried 2.4b and 7.8b both thinking and instruct. They generate way worse results than `gemma3` and `qwen3` models and often produce wrong number of exercises.

After Ollama is set up you can use it programmatically or via the CLI + configuration file (see below).

## Configuration file

The configuration uses TOML format. Available fields:

| Parameter            | Description                                                                                                                                                                                                                                                                                                                                                                                                                             | Example                                                |
|----------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------|
| `topics_folder`      | Location of the folder containing descriptions of exercise topics.                                                                                                                                                                                                                                                                                                                                                                      | `"hmeg/topics"`                                        |
| `vocab_file`         | Location of the vocabulary file, which will be used for generation of exercises. Required for the "templates" engine.                                                                                                                                                                                                                                                                                                                   | `"hmeg/vocabs/minilex.toml"`                           |
| `vocab_level`        | Optional. CEFR level for vocabulary selection: A1, A2, B1, B2, C1, C2. Used by the "ollama" engine.                                                                                                                                                                                                                                                                                                                                     | `"B2"`                                                 |
| `topic`              | Name of the topic for generation of exercises. Can be partial (see CLI instructions above).                                                                                                                                                                                                                                                                                                                                             | `"Have, Don’t have, There is, There isn’t / 있어요, 없어요"` |
| `number_exercises`   | Number of generated exercises (5-100).                                                                                                                                                                                                                                                                                                                                                                                                  | `15`                                                   |
| `engine`             | Exercise generation engine. Can be "templates" or "ollama".                                                                                                                                                                                                                                                                                                                                                                             | `"ollama"`                                             |
| `model`              | Name of the LLM model for Ollama. Must be defined if `engine` is set to "ollama".                                                                                                                                                                                                                                                                                                                                                       | `"gemma3:4b"`                                          |
| `grammar_correction` | Optional. Defines the model used for grammar correction in exercises generated via the "templates" engine. Experimental. Supported models:<br>* `"kenlm/en"` -- KenLM-based model. Requires files `en.arpa.bin`, `en.sp.model`, `en.sp.vocab` in the `lm` folder.<br>* `distilbert/distilgpt2` -- Distilled-GPT2 model from HuggingFace.<br>* `openai` -- one of OpenAI's models. Defined in the `hmeg/prompts/v1/reranker/openai.yaml` | `"kenlm/en"`                                           |

Notes:
* Miniphrase exercises are supported only when using the "templates" engine.
* When using the `"openai"` reranker, create a `.env` file in the project root directory (the same directory
as `hmeg_cli.py`) and set the `OPENAI_API_KEY` variable. You can use the provided `.env.template` file as a
starting point.

### Configuration example for "templates" engine

```toml
topics_folder="hmeg/topics"
vocab_file="hmeg/vocabs/minilex.toml"

topic="Have, Don’t have, There is, There isn’t / 있어요, 없어요"
number_exercises=15

engine="templates"
grammar_correction="kenlm/en"
```

### Configuration example for "ollama" engine

```toml
topics_folder="hmeg/topics"
vocab_level="C1"

topic="Have, Don’t have, There is, There isn’t / 있어요, 없어요"
number_exercises=15

engine="ollama"
model="gemma3:4b"
```

# Format of exercises and vocabulary

The library supports extensible templates for exercise generation and customizable vocabulary.

Built-in exercises topics and vocabulary can be found in
[hmeg/topics/](hmeg/topics/) and [hmeg/vocabs/minilex.toml](hmeg/vocabs/minilex.toml) 

See the [docs](docs) folder for details on the format for exercises and vocabulary.

# Why I made this library

A few words about the name: Erik Gunnemark was a pre-internet hyperpolyglot who translated from
more than 20 languages. He co-authored "The Art and Science of Learning Languages". The book introduces
the idea of a Minilex -- a few hundred core words that cover many situations.

I created this library to provide speaking drills focused on small, simple grammatical structures
and a limited vocabulary. Compared to exercises generated by large language models, these exercises
are simpler and rely on a controlled vocabulary that can be expanded. The templates are editable,
and the dictionary can be swapped to suit different goals (e.g.,
[Basic English](https://en.wikipedia.org/wiki/Basic_English) or domain-specific vocabularies).

Lastly, the project name is a light Star Wars reference :)

UPD (2026.01.24): In the last 2 years, there has been good progress in LLM abilities to process
Korean language. Therefore, a support for LLM-based generation of exercises has been added via Ollama.
It has less control over vocabulary and structures of exercises, but can generate more natural sentences.
Vocabulary can be controlled to some extent by using CEFR levels (A1-C2).
