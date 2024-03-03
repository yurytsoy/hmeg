import sys
import tomllib


if len(sys.argv) > 1:
    config_file = sys.argv[1]
else:
    config_file = 'hmeg.conf'


with open(config_file, mode="r") as f:
    config = tomllib.loads(f.read())

print("topic:", config["topic"])
print("# exercises:", config["number_exercises"])
