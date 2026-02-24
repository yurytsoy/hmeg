# hmeg -- speaking and translation exercises generator

[![Unit-tests](https://github.com/yurytsoy/hmeg/actions/workflows/tests.yml/badge.svg)](https://github.com/yurytsoy/hmeg/actions/workflows/tests.yml)

Help me, Erik Gunnemark -- a library for generating exercises to practice basic speaking constructs.

## Table of Contents

- [Installation](#installation)
  - [Via pip](#via-pip)
  - [Via git](#via-git)
- [Usage](#usage)
  - [Command line](#command-line)
  - [Python](#python)
  - [Generating exercises using Ollama](#generating-exercises-using-ollama)
- [Format of exercises and vocabulary](#format-of-exercises-and-vocabulary)
- [Why I made this library](#why-i-made-this-library)

The idea is that mastering these building blocks helps with faster speaking and constructing more complex sentences.

Exercises are generated either based on templates or using large language models (LLMs) via Ollama.
The "templates" engine provides more control over the structure and vocabulary of exercises, while the "ollama" engine
can generate more natural sentences with less control.

Interactive tutor mode is available via the CLI, which uses an AI agent to guide you through exercises and provide
feedback on your translations. The tutor relies on local LLMs, so the quality of feedback may vary.

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

## Command line

The CLI tool `hmeg` is available after installation.

### Interactive tutor (AI Agent)

You can run tutor using
```bash
hmeg tutor
```

The tutor requires local installation of [Ollama](https://ollama.com/download). The tutor will guide you through exercises and provide feedback on your translations.

Tutor configuration is taken from the file `tutor.conf`. Example can be found in the file [tutor.conf.template](tutor.conf.template)

Models that support tools are recommended for the best experience. Some recommended models:
* `gpt-oss:20b` -- pretty good and powerful model. Better works with 16GB of VRAM.
* `qwen3:4b-instruct` -- fast small model with decent Korean. Regular `qwen3:4b` is waaaay slower.

> **Important**: Local models make mistakes sometimes, so the tutor's feedback may not always be accurate.
> Use it as a guide, but trust your own judgment as well!

Example session with the tutor (gpt-oss-20b):

```
Hello! 👋 How would you like to practice today?                                                                                                                                                                                             

 • A specific Korean grammar point (e.g., “-아/어/여도”, “‑고 싶다”, “‑는 걸요”)?                                                                                                                                                           
 • Or a general CEFR level (A2, B1, etc.) for translation exercises?                                                                                                                                                                        

Let me know what you’d like to work on!                                                                                                                                                                                                     
You: please give me a random exercise for B1
Here’s a B1‑level sentence for you to translate into Korean:                                                                                                                                                                                

▌ I want to go to the movie theater.                                                                                                                                                                                                      

Give it a try and let me know your translation!                                                                                                                                                                                             
You: 저는 영화관에 가고 싶어요.
Great job! 👏                                                                                                                                                                                                                               
Your sentence 저는 영화관에 가고 싶어요 is correct.                                                                                                                                                                                         
Just a tiny note: “가고 싶어요” is a polite form; if you want to match a more formal style, you could use 가고 싶습니다. Keep it up!                                                                                                        
You: I am good for today
Finishing session...
It was a pleasure helping you! 🎉                                                                                                                                                                                                           
Feel free to return anytime you want to practice more. Have a wonderful day!                                                                                                                                                                
Session finished by agent.
```

### Classical CLI

Update file [hmeg.conf](hmeg.conf) to select the grammatical topic and number of exercises.
More details about configuration can be found in the [docs](docs/cli_configuration.md).

Run:
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

## Generating exercises using Ollama

You can use [Ollama](https://ollama.com/) to generate exercises. Follow the official install instructions for your platform.

Recommended models:
* `gemma3` -- `4b` and `12b` work pretty well comparing to other families that I tried.
* `qwen3` -- `4b-instruct` is also not bad and much (much) faster than the thinking variant.

Note on `exaone3.5` (2026.01.24): I had high hopes, since the models were prepared by LG. Tried 2.4b and 7.8b both thinking and instruct. They generate way worse results than `gemma3` and `qwen3` models and often produce wrong number of exercises.

After Ollama is set up you can use it programmatically or via the CLI + configuration file (see below).

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
