import dataclasses


@dataclasses.dataclass
class GrammarDescription:
    name: str
    links: list[str]
    exercises: list[str]
