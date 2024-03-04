import sys
import tomllib

from hmeg import utils, ExerciseGenerator


if len(sys.argv) > 1:
    config_file = sys.argv[1]
else:
    config_file = 'hmeg.conf'


with open(config_file, mode="r") as f:
    config = tomllib.loads(f.read())


utils.register_grammar_topics(config["topics_folder"])

ExerciseGenerator.generate_exercises(topic_name=config["topic"], num=config["number_exercises"])
