# Vocabulary

## Overview

Vocabulary contains list of words that are going to be used during generation
of exercises.

Vocabulary can be extended or replaced with user-defined one in order to
focus on specific topic or extend list of used words.

Example of the vocabulary can be found in the file [hmeg/vocabs/minilex.toml](hmeg/vocabs/minilex.toml).

## Vocabulary description format

A vocabulary is described in a toml-file with the following sections:
* `name` -- name of the vocabulary
* `verbs` -- list of verbs in present simple
* `nouncs` -- list of singular nouns
* `adjectives` -- list of adjectives
* `adverbs` -- list of adverbs

## Using `Vocabulary` class

`Vocabulary` class is responsible for loading the vocabulary contents
and using those in order to generate random words for exercises. 

* Loading:
```python
from hmeg import Vocabulary

vocab = Vocabulary.load("hmeg/vocabs/minilex.toml")
```
* Getting random word from a loaded vocabulary:
```python
vocab.random_verb()
vocab.random_noun()
vocab.random_adverb()
vocab.random_adjective()
vocab.random_weekday()
vocab.random_season()
```