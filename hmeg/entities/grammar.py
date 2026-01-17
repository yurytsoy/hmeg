from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class TopicLevelInfo:
    """
    Describes the level information associated with a single grammar resource.

    Attributes:
        resource_name: Identifier or name of the grammar resource (e.g. topic or lesson).
        level: Proficiency level for the resource. Can be an integer level, a string
            descriptor (such as "A1", "B2", or "intermediate"), or None if no level
            is specified.
    """

    resource_name: str
    level: int | str | None = None


@dataclasses.dataclass
class GrammarDescription:
    """
    Representation of a grammar topic with associated resources.

    Attributes:
        name: The canonical name of the grammar topic.
        links: A list of resource URLs or identifiers related to the topic.
        exercises: A list of exercise identifiers or descriptions for practice.
        levels: A list of `TopicLevelInfo` items describing level-specific resources.

    Methods:
        from_dict: Construct a `GrammarDescription` from a mapping (typically loaded from YAML/JSON).
    """

    name: str
    links: list[str]
    exercises: list[str]
    levels: list[TopicLevelInfo]

    @staticmethod
    def from_dict(d: dict) -> GrammarDescription:
        res = GrammarDescription(
            name=d["name"],
            links=d["links"],
            exercises=d["exercises"],
            levels=[TopicLevelInfo(**level_descr) for level_descr in d.get("levels", [])],
        )
        return res
