from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class TopicLevelInfo:
    resource_name: str
    level: int | str | None = None


@dataclasses.dataclass
class GrammarDescription:
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
