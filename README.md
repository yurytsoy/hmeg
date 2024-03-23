# hmeg

Help me, Erik Gunnemark -- library for generation of exercises for practicing of some basic Korean speaking blocks.
The (very hopeful) idea is that mastering these blocks can help with faster speaking and constructing more
complex sentences. 

The exercises are generated randomly and thus can occasionally come out
as grammatically strange or even inappropriate. In either case, if you use this library and
encounter such sentences please let me know.

But as long as the sentence is not abusive and grammatically correct I assume
that it is a proper exercise, with the goal is to practice quickfire translation into Korean,
where element of surprise can facilitate better memorization.

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
