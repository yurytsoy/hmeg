from .exercise import TranslationExercise
from .grammar import GrammarDescription, TopicLevelInfo
from .prompt import Prompt
from .vocab import VocabularyInfo, VocabularyPlaceholders

VOWELS = "aeiou"


class ExerciseGenerationEngine:
    """
    An enumeration of exercise generation engines.
    """
    LEGACY = "legacy"  # template-based generation
    OLLAMA = "ollama"
