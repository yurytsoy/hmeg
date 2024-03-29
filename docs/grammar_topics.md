# Grammar topics

## Overview

Exercises are generated depending on the grammar topic. Each grammar topic is assumed to be stored in
a separate toml-file and can define exercise template corresponding for the topic.

The topics are addressed and identified by the name.
Topics are loaded before generation of exercises depending on the `topics_folder` variable in the configuration file.

Vocabulary for exercises for topic is defined in the `vocab_file` variable in configuration.

See folder `hmeg/topics/` for examples.

## Topic description format

Each grammar topic is defined by a structure, that includes sections:
* `name` -- topic name
* `levels` -- list of study level for different resources and textbooks. Can be helpful to understand the complexity and for reference.
* `links` -- links to the resources with explanation
* `exercises` -- list of NLTK-compatible templates that describe structure of generated exercises.
    Each template is essentially a structure in a [Backus-Naur form](https://en.wikipedia.org/wiki/Backus%E2%80%93Naur_form). 
    When several different templates are defined, they are picked at random for generation of a next exercise.
    * Apostrophes: use backtick "\`" (`~` key) instead of a single quote "'" 

```toml
name="While / -(으)면서"
levels=[
    {resource_name="TTMIK", level=9},
    {resource_name="King Sejong Institute Practical Korean", level=3}
]
links=[]
exercises=[
    """
    S -> P V 'and' V
    P -> 'I' | 'they' | 'you' | 'we' |
    V -> '{verb}'
    """,
    """
    S -> P V 'and' V
    P -> 'he' | 'she' | 'it' |
    V -> '{verb:3s}'
    """
]
```

## Grammar token-nodes

In order to make generated sentences extensible and more compact, the grammar can include special token-nodes
that represent group of words, instead of a specific word. Those tokens are replaced with random words of
corresponding quality during generation of exercises.

The following token-nodes are allowed:

| Token             | Description                                                 | Examples                                                                     |
|-------------------|-------------------------------------------------------------|------------------------------------------------------------------------------|
| `{adj}`           | adjective                                                   | "calm", "famous", "light"                                                    |
| `{adverb}`        | adverb                                                      | "importantly", "relatively", "strangely"                                     |
| `{city}`          | name of a city                                              | "Paris", "Seoul", "Berlin"                                                   |
| `{country}`       | name of a country                                           | "France", "Korea", "Germany"                                                 |
| `{month}`         | name of a month                                             | "January", "February", "March", ...                                          |
| `{nationality}`   | name of a nationality                                       | "French", "Korean", "German"                                                 |
| `{noun}`          | regular singular noun                                       | "word", "house", "tree"                                                      |
| `{a:noun}`        | regular singular noun with a preceding "a" or "an" article  | "a word", "a house", "an apple"                                              |
| `{number:100}`    | number from 0 to 99                                         | "96", "54", "32"                                                             |
| `{number:1000}`   | number from 0 to 999                                        | "96", "754", "342"                                                           |
| `{number:100000}` | number from 0 to 99,999                                     | "96", "75,489", "3,424"                                                      |
| `{place}`         | name of a place                                             | "apartment", "work", "airport"                                                |
| `{verb}`          | regular verb in present simple                              | "go", "work", "see"                                                          |
| `{verb:3s}`       | regular verb in the present simple for 3rd person, singular | "goes", "works", "sees"                                                      |
| `{verb:ing}`      | regular verb in the present continuous tense                | "going", "working", "seeing"                                                 |
| `{verb:past}`     | regular verb in the past simple tense                       | "went", "worked", "saw"                                                      |
| `{season}`        | time of the year                                            | "Spring", "Summer", "Autumn", "Winter"                                       |
| `{weekday}`       | day of the week                                             | "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday" |
