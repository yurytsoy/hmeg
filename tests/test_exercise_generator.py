import os
import random
import unittest

from pydantic_ai.messages import ModelResponse, ToolCallPart

from hmeg import GrammarRegistry, usecases, ExerciseGenerator
from hmeg.entities import VocabularyPlaceholders
from hmeg.usecases import is_ollama_available

TEST_OLLAMA_MODEL = "gemma4:e2b"
os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434/v1"


class TestExerciseGenerator(unittest.TestCase):
    @classmethod
    def setUp(cls):
        super().setUp(cls)
        usecases.register_grammar_topics("hmeg/topics/")

    def test_generate_exercises(self):
        for topic in GrammarRegistry.topics:
            exercises = ExerciseGenerator.generate_exercises(topic, num=5)
            self.assertEqual(len(exercises), 5)

            # check that no exercises contain placeholders.
            for placeholder in VocabularyPlaceholders.to_list():
                self.assertTrue(all(placeholder not in res for res in exercises))

    def test_generate_exercises_unregistered_topic(self):
        with self.assertRaises(RuntimeError):
            ExerciseGenerator.generate_exercises("bad topic", num=10)

    @unittest.skipIf(
        not is_ollama_available(TEST_OLLAMA_MODEL),
        "Run locally with Ollama service available"
    )
    def test_generate_exercises_ollama(self):
        from hmeg.exercise_generator.ollama_gen import generate_exercises

        random.seed(42)
        topics = random.choices(list(GrammarRegistry.topics), k=5)

        for topic in topics:
            exercises = generate_exercises(topic, num=5, model=TEST_OLLAMA_MODEL)
            self.assertEqual(len(exercises), 5)
            print(f"Exercises for topic '{topic}':")
            for ex in exercises:
                print(f"- {ex.sentence_kr} / {ex.sentence_en}")

            # check that no exercises contain placeholders.
            for placeholder in VocabularyPlaceholders.to_list():
                self.assertTrue(all(placeholder not in res.sentence_en for res in exercises if res.sentence_en is not None))
                self.assertTrue(all(placeholder not in res.sentence_kr for res in exercises if res.sentence_kr is not None))

    @unittest.skipIf(
        not is_ollama_available(TEST_OLLAMA_MODEL),
        "Run locally with Ollama service available"
    )
    def test_generate_exercises_ollama_with_vocabulary_level(self):
        from hmeg.exercise_generator.ollama_gen import generate_exercises

        random.seed(37)
        topic = random.choices(list(GrammarRegistry.topics), k=1)[0]
        print(f"Exercises for topic '{topic}':")

        exercises_a1 = generate_exercises(topic, num=5, model=TEST_OLLAMA_MODEL, vocab_level="A1")
        max_len_kr_a1 = max(len(ex.sentence_kr) for ex in exercises_a1)
        max_len_en_a1 = max(len(ex.sentence_en) for ex in exercises_a1)
        for ex in exercises_a1:
            print(f"- [A1] {ex.sentence_kr} / {ex.sentence_en}")

        exercises_c2 = generate_exercises(topic, num=5, model=TEST_OLLAMA_MODEL, vocab_level="C2")
        max_len_kr_c2 = max(len(ex.sentence_kr) for ex in exercises_c2)
        max_len_en_c2 = max(len(ex.sentence_en) for ex in exercises_c2)
        for ex in exercises_c2:
            print(f"- [C2] {ex.sentence_kr} / {ex.sentence_en}")

        # more advanced vocabulary level normally leads to longer and more complex sentences.
        # but allow for some variance in length due to model randomness.
        self.assertGreaterEqual(max_len_kr_c2 + 8, max_len_kr_a1)
        self.assertGreaterEqual(max_len_en_c2 + 8, max_len_en_a1)

    @unittest.skipIf(
        not is_ollama_available(TEST_OLLAMA_MODEL),
        "Run locally with Ollama service available"
    )
    def test_pydantic_generator(self):
        from hmeg.exercise_generator.pydantic_gen import generate_exercises

        random.seed(42)
        topics = random.choices(list(GrammarRegistry.topics), k=1)
        exercises = generate_exercises(topics[0], num=40, verbose=True)
        print(topics[0])
        print("Generated exercises:", len(exercises))
        print(exercises)

    @unittest.skipIf(
        not is_ollama_available(TEST_OLLAMA_MODEL),
        "Run locally with Ollama service available"
    )
    def test_eval_agent(self):
        from hmeg.exercise_generator.pydantic_gen.generator import make_evaluator_agent, get_evaluator_prompt, read_all_lines

        random.seed(42)
        correct_topic, incorrect_topic = random.choices(list(GrammarRegistry.topics), k=2)
        correct_cefr, incorrect_cefr1, incorrect_cefr2 = "A2", "B2", "A1"

        model_name = "gemma4:e4b"
        agent = make_evaluator_agent(model_name=model_name)
        input_filename = "tests/data/result_10.txt"

        with self.subTest("Correct topic & CEFR"):
            agent.run_sync(get_evaluator_prompt(
                topic_name=correct_topic, vocab_level="A2", input_filename=input_filename, out_filename="result_corr_A2.txt")
            )

        with self.subTest("Correct topic, incorrect CEFR"):
            for vocab_level in [incorrect_cefr1, incorrect_cefr2]:
                agent.run_sync(get_evaluator_prompt(
                    topic_name=correct_topic,
                    vocab_level=vocab_level,
                    input_filename=input_filename,
                    out_filename=f"result_corr_{vocab_level}.txt"
                ))
            self.assertLess(os.stat(f"result_corr_A1.txt").st_size, os.stat(f"result_corr_A2.txt").st_size)
            self.assertEqual(os.stat(f"result_corr_A2.txt").st_size, os.stat(f"result_corr_B1.txt").st_size)

        with self.subTest("Incorrect topic, correct CEFR"):
            agent.run_sync(get_evaluator_prompt(
                topic_name=incorrect_topic, vocab_level="A2", input_filename=input_filename, out_filename="result_incorr_A2.txt")
            )
            self.assertFalse(not os.path.exists("result_incorr_A2.txt"))  # nothing should be passed through.

    @unittest.skipIf(
        not is_ollama_available(TEST_OLLAMA_MODEL),
        "Run locally with Ollama service available"
    )
    def test_gen_agent(self):
        from hmeg.exercise_generator.pydantic_gen.usecases import make_generator_agent

        random.seed(42)
        topics = random.choices(list(GrammarRegistry.topics), k=2)

        model_name = "gemma4:e4b"
        agent = make_generator_agent(model_name=model_name)

        for topic in topics:
            resp = agent.run_sync(f"Generate 5 exercises for the topic: {topic} for vocabulary A1. Write resulting phrases into the file 'result.txt', each exercise in a separate line.")
            print(topic)
            print(resp.output)
            print(resp.usage)

            tool_index = 1
            for message in resp.new_messages():
                # Tool call instructions from the LLM always arrive inside ModelResponse objects
                if isinstance(message, ModelResponse):
                    for part in message.parts:
                        # Check if this part of the response is an explicit tool call
                        if isinstance(part, ToolCallPart):
                            print(f"[{tool_index}] Tool Called: {part.tool_name}")
                            print(f"    - Parameters: {part.args}")
                            print(f"    - Execution ID: {part.tool_call_id}")
                            print("-" * 40)
                            tool_index += 1
