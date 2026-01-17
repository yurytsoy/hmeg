# hmeg -- speaking and translation exercises generator

[![Unit-tests](https://github.com/yurytsoy/hmeg/actions/workflows/tests.yml/badge.svg)](https://github.com/yurytsoy/hmeg/actions/workflows/tests.yml)

Help me, Erik Gunnemark -- a library for generating exercises to practice basic speaking constructs.

The idea is that mastering these building blocks helps with faster speaking and constructing more complex sentences.

Exercises are generated randomly, so they can sometimes be grammatically or semantically odd.
As long as a sentence is not abusive and is grammatically correct, it is considered a valid exercise.
The goal is to facilitate quickfire translation into Korean, where the element of surprise can aid memorization.

# Usage

## Command line

Update file [hmeg.conf](hmeg.conf) to select the grammatical topic and number of exercises,
then run:
```bash
python hmeg_cli.py
```

You can also specify command-line arguments to define configuration file, topic, and/or number of generated exercises.

* Run with a custom configuration file (use the `run` subcommand):
```bash
python hmeg_cli.py run --config="custom/configuration/file.toml"
```

* Run with a custom topic and number of exercises:
```bash
python hmeg_cli.py run -n 15 -t "Have, Don’t have, There is, There isn’t / 있어요, 없어요"
```

* You can provide a partial topic name. All topics that contain the specified string will be used:
```bash
python hmeg_cli.py run -n 15 -t "있어요, 없어요"
python hmeg_cli.py run -n 15 -t "there is"
```

* List available topics described in the specified configuration file:
```bash
python hmeg_cli.py list -c hmeg.conf
```

* Print help:

```bash
python hmeg_cli.py --help
python hmeg_cli.py run --help
python hmeg_cli.py list --help
```

### Configuration file

The configuration uses TOML format. Available fields:

| Parameter | Description                                                                                                                                                                                                                                                                                                                                                                                                | Example                                                |
|-----------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------|
| `topics_folder` | Location of the folder containing descriptions of exercise topics.                                                                                                                                                                                                                                                                                                                                         | `"hmeg/topics"`                                        |
| `vocab_file` | Location of the vocabulary file, which will be used for generation of exercises.                                                                                                                                                                                                                                                                                                                           | `"hmeg/vocabs/minilex.toml"`                           |
| `topic` | Name of the topic for generation of exercises. Can be partial (see CLI instructions above).                                                                                                                                                                                                                                                                                                                | `"Have, Don’t have, There is, There isn’t / 있어요, 없어요"` |
| `number_exercises` | Number of generated exercises (5-100).                                                                                                                                                                                                                                                                                                                                                                     | `15` |
| `grammar_correction` | Optional. Defines the model used for grammar correction in generated exercises. Experimental. Supported models:<br>* `"kenlm/en"` -- KenLM-based model. Requires files `en.arpa.bin`, `en.sp.model`, `en.sp.vocab` in the `lm` folder.<br>* `distilbert/distilgpt2` -- Distilled-GPT2 model from HuggingFace.<br>* `openai` -- one of OpenAI's models. Defined in the `hmeg/prompts/v1/reranker/openai.yaml` | `"kenlm/en"`                                           |

Example (`hmeg.conf`):
```toml
topics_folder="hmeg/topics"
vocab_file="hmeg/vocabs/minilex.toml"

topic="Have, Don’t have, There is, There isn’t / 있어요, 없어요"
number_exercises=15

grammar_correction="kenlm/en"
```
Notes:
* When using the `"openai"` reranker, create a `.env` file in the project root directory (the same directory
as `hmeg_cli.py`) and set the `OPENAI_API_KEY` variable. You can use the provided `.env.template` file as a
starting point.


## Python code

```python
from hmeg import utils, ExerciseGenerator, load_minilex


num_exercises = 10  # number of randomly generated exercises for the selected topic

utils.register_grammar_topics()
vocab = load_minilex()  # load words from the Minilex.

exercises = ExerciseGenerator.generate_exercises(
    topic_name="While / -(으)면서", num=num_exercises, vocab=vocab
)
print("\n".join(exercises))
```

# Format of exercises and vocabulary

The library supports extensible templates for exercise generation and customizable vocabulary.

Built-in exercises topics and vocabulary can be found in
[hmeg/topics/](hmeg/topics/) and [hmeg/vocabs/minilex.toml](hmeg/vocabs/minilex.toml) 

See the [docs](docs) folder for details on the format for exercises and vocabulary.

# Why I made this library

A few words about the name: Erik Gunnemark was a pre-internet hyperpolyglot who translated from
more than 20 languages. He co-authored The Art and Science of Learning Languages. The book introduces
the idea of a Minilex -- a few hundred core words that cover many situations.

I created this library to provide speaking drills focused on small, simple grammatical structures
and a limited vocabulary. Compared to exercises generated by large language models, these exercises
are simpler and rely on a controlled vocabulary that can be expanded. The templates are editable,
and the dictionary can be swapped to suit different goals (e.g.,
[Basic English](https://en.wikipedia.org/wiki/Basic_English) or domain-specific vocabularies).

Lastly, the project name is a light Star Wars reference :)
