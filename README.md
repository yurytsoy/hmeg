# hmeg
Help me, Erik Gunnemark

# Grammar topics

Each grammar topic is defined by a structure, that includes sections:
* `name` -- topic name
* `levels` -- list of study level for different resources and textbooks. Can be helpful to understand the complexity and for reference.
* `links` -- links to the resources with explanation
* `exercises` -- list of NLTK-compatible grammars that describe structure of generated exercises. Can include several different structures.

```python
description = {
    "name": "-으면서 / -면서",
    "levels": [
        {"resource_name": "TTMIK", "level": 9},
        {"resource_name": "King Sejong Institute Practical Korean", "level": 3}
    ],
    "links": [],
    "exercises": [
        """
        S -> P V 'and' V
        P -> 'I' | 'they' | 'you' | 'we' |
        V -> '{verb}'
        """,
        """
        S -> P V 'and' V
        P -> 'he' | 'she' | 'it' | 
        V -> '{verb:s}'
        """
    ]
}
```

## Grammar token-nodes

In order to make generated sentences extensible and more compact, the grammar can include special token-nodes
that represent group of words, instead of a specific word.

The following token-nodes are allowed:
* `{noun}` -- regular singular noun
   * "word", "house", "tree", ... 
* `{verb}` -- regular verb
   * "go", "work", "see", ... 
* `{verb:3s}` -- regular verb in the present simple for 3rd person, singular
   * "goes", "works", "sees", ... 
* `{pronoun}` -- personal pronouns
   * "I", "we", "they", "you" 
* `{pronoun:3s}` -- personal pronouns for 3rd person, singular
   * "he", "she", "it" 
