import sys
import tomllib

from hmeg import utils, ExerciseGenerator, Vocabulary


if len(sys.argv) > 1:
    config_file = sys.argv[1]
else:
    config_file = 'hmeg.conf'


with open(config_file, mode="r") as f:
    config = tomllib.loads(f.read())


utils.register_grammar_topics(config["topics_folder"])

vocab = Vocabulary.load(config["vocab_file"])
exercises = ExerciseGenerator.generate_exercises(
    topic_name=config["topic"], num=config["number_exercises"], vocab=vocab
)
print(f"Exercises for topic: {config['topic']}")
for idx, exercise in enumerate(exercises):
    print(f"{idx+1}. {exercise}")
