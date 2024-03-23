# hmeg

Help me, Erik Gunnemark -- library for generation of exercises for practicing of some basic speaking blocks.

TODO: who was Erik Gunnemark.

TODO: why I made this library, its purpose and usage.

TODO: disclaimer about randomness of generated exercises and them not making much sense.

# Usage

```python
from hmeg import utils, Vocabulary, ExerciseGenerator

topics_folder = "hmeg/topics/"  # folder containing description of exercises for different grammar topics.
vocab_file = "vocabs/minilex.toml"  # file with vocabulary for generation of exercises.
num_exercises = 10  # number of randomly generated exercises for selected topic

utils.register_grammar_topics(topics_folder)
vocab = Vocabulary.load(vocab_file)

exercises = ExerciseGenerator.generate_exercises(
    topic_name="While / -(으)면서", num=num_exercises, vocab=vocab
)
print(exercises)
```

# Format of exercises and vocabulary

The library supports extensible and configurable templates for generation of exercises
as well as customizable vocabulary.

The built-in exercises topics and vocabulary can be found in the `hmeg/topics/` and `vocabs/minilex.toml` 

See folder `docs` for more details regarding description of format for exercises and vocabulary.
