# Configuration file (classical CLI)

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

## Configuration example for "templates" engine

```toml
topics_folder="hmeg/topics"
vocab_file="hmeg/vocabs/minilex.toml"

topic="Have, Don’t have, There is, There isn’t / 있어요, 없어요"
number_exercises=15

engine="templates"
grammar_correction="kenlm/en"
```

## Configuration example for "ollama" engine

```toml
topics_folder="hmeg/topics"
vocab_level="C1"

topic="Have, Don’t have, There is, There isn’t / 있어요, 없어요"
number_exercises=15

engine="ollama"
model="gemma3:4b"
```

