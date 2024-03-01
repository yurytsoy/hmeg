description = {
    "name": "-으면서 / -면서",
    "links": [],
    "grammars": [
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
