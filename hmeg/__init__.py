import logging

class LambdaWarningFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "Accessing `LambdaRuntimeClient`" not in record.getMessage()

logging.getLogger("transformers").addFilter(LambdaWarningFilter())


from .exercise_generator import ExerciseGenerator
from .grammar_checker import GrammarChecker
from .language_tool_manager import LanguageToolManager
from .grammar_registry import GrammarRegistry
from .reranker import Reranker
from .vocabulary import Vocabulary, load_minilex
