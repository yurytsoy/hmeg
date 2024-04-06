# hmeg

Help me, Erik Gunnemark -- library for generation of exercises for practicing of some basic speaking constructs.
The (very hopeful) idea is that mastering these blocks can help with faster speaking and constructing more
complex sentences. 

The exercises are generated randomly and thus can occasionally come out
as grammatically or semantically strange.

But as long as the sentence is not abusive and grammatically correct I assume
that it is a proper exercise. With the goal to practice quickfire translation into Korean,
where element of surprise can facilitate better memorization.

# Usage

## Command line

Update file [hmeg.conf](hmeg.conf) to select the grammatical topic and number of exercises
and then execute:
```bash
python hmeg_cli.py
```

You can also specify command line arguments to define configuration file, topic,
and or number of generated exercises.

* Run with custom configuration file (notice the `run` keyword)
```bash
python hmeg_cli.py run --config="custom/configuration/file.toml"
```

* Run with custom topic and number of exercises
```bash
python hmeg_cli.py run -n 15 -t "Have, Don’t have, There is, There isn’t / 있어요, 없어요"
```

* You can use partial name of the topic. In that case all topics, that contain the specified string will be used.
```bash
python hmeg_cli.py run -n 15 -t "있어요, 없어요"
python hmeg_cli.py run -n 15 -t "there is"
```

* List available topics described in the specified configuration file (optional)
```bash
python hmeg_cli.py list -c hmeg.conf
```

* Print help for the arguments or specific command

```bash
python hmeg_cli.py --help
python hmeg_cli.py run --help
python hmeg_cli.py list --help
```

## Python code

```python
from hmeg import utils, ExerciseGenerator, load_minilex


num_exercises = 10  # number of randomly generated exercises for selected topic

utils.register_grammar_topics()
vocab = load_minilex()  # load words from the Minilex.

exercises = ExerciseGenerator.generate_exercises(
    topic_name="While / -(으)면서", num=num_exercises, vocab=vocab
)
print("\n".join(exercises))
```

# Format of exercises and vocabulary

The library supports extensible and configurable templates for generation of exercises
as well as customizable vocabulary.

The built-in exercises topics and vocabulary can be found in the
[hmeg/topics/](hmeg/topics/) and [hmeg/vocabs/minilex.toml](hmeg/vocabs/minilex.toml) 

See folder [docs](docs) for more details regarding description of format for exercises and vocabulary.

# Why I made this library

First of all, few words about the naming of this library.

Erik Gunnemark was a pre-Internet era hyper-polyglot, who was able to translate from >20 languages.
He is an author of the book "The Art and Science of Learning Languages" co-authored with Amorey Gethin.
Among other things the book contains idea of a Minilex,
a few hundred core words, picked based on the author's experience with learning multiple languages.
Those words are considered to be important and cover a lot of situations.

As I am having lots of struggles with learning to speak Korean (>10 years), I was
thinking about having additional exercises for speaking drills. And
being a fan of the "crawling before walking" I was thinking that in
order to speak full-length sentences, the smaller and simpler
grammatical structures and phrases need to be internalized first.

I tried ChatGPT for getting exercises, but it produced quite repetitive
output. And it was harder for me to control structure of exercises
and restrict the vocabulary. So I decided to make my own exercise
generator with flexible vocabulary and exercises.

Additionally, normally we do not need to use too many words to express
ourselves so that our passive vocabulary much larger than the active one.
Meaning that it can be ok to focus on a proper usage
of core words and grammatical structures. But proper choice of words
to rely upon is not that simple. This is where Minilex might be useful as it
is based on the experience of the professional language learners.

However, Minilex is not the only option. Therefore, library was designed in such a way
that dictionary can be switched depending on the user's goals.
A good alternative to Minilex could be [Basic English](https://en.wikipedia.org/wiki/Basic_English)
or specialized subset of words eg for medicine or engineering.  

Lastly, the library naming is a plain reference to Star Wars :)
