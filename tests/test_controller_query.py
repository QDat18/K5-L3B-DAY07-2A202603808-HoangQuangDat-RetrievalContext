import sys
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'starter'))
# agent.py import retriever nặng (sentence_transformers/psycopg); test này không cần.
sys.modules.setdefault('retrievers', types.SimpleNamespace(
    DenseRetriever=None, LexicalRetriever=None, RerankRetriever=None))
import agent  # noqa: E402


class FirstQueryTests(unittest.TestCase):
    def test_first_turn_uses_original_question_without_llm_call(self):
        c = agent.NimController(api_key='dummy')
        c.client = mock.Mock()
        q = 'Hạ đường huyết mức 2 được xác định khi glucose huyết ở ngưỡng nào?'
        self.assertEqual(c.decide(q, []), {'action': 'search', 'query': q})
        c.client.chat.completions.create.assert_not_called()


class CitationFormatTests(unittest.TestCase):
    def test_bracketed_citation_ids_are_normalized(self):
        # q03/q06: model trả "citations": ["[B-1]"] -> trước đây bị coi là không hợp lệ -> mất câu trả lời
        c = agent.NimController(api_key='dummy')
        c._verify_support = lambda q, a, e: True
        c._chat = lambda prompt, max_tokens, retries=2, **kw: (
            '{"action": "answer", "answer": "Ăn nhẹ ngay [B-1].", "citations": ["[B-1]", " B-2 "]}')
        ctx = '[B-1 | d | page 1]\nĂn nhẹ ngay.\n\n[B-2 | d | page 1]\nTheo dõi.'
        r = c.decide('q?', [{'context': ctx, 'context_ids': ['B-1', 'B-2']}])
        self.assertEqual(r['citations'], ['B-1', 'B-2'])
        self.assertIn('[B-1]', r['answer'])


class VerifierQuoteTests(unittest.TestCase):
    def _controller(self, reply):
        c = agent.NimController(api_key='dummy')
        c._chat = lambda prompt, max_tokens, retries=2, **kw: reply
        return c

    def test_supported_true_but_fabricated_quote_is_rejected(self):
        c = self._controller('{"main_point": "x", "quote": "câu không hề có trong đoạn trích", "supported": true}')
        self.assertFalse(c._verify_support('q?', 'a', 'Đoạn trích thật sự nói về một chuyện khác hoàn toàn.'))

    def test_supported_true_with_real_quote_passes(self):
        c = self._controller('{"main_point": "x", "quote": "nên đo 6 lần/ngày", "supported": true}')
        self.assertTrue(c._verify_support('q?', 'a', 'BN điều trị insulin nên đo 6 lần/ngày vào trước và sau ăn.'))

    def test_supported_false_stays_false(self):
        c = self._controller('{"main_point": "x", "quote": "nên đo 6 lần/ngày", "supported": false}')
        self.assertFalse(c._verify_support('q?', 'a', 'nên đo 6 lần/ngày'))


class ApiRetryTests(unittest.TestCase):
    def _ok(self, text):
        r = mock.Mock()
        r.choices = [mock.Mock(finish_reason='stop', message=mock.Mock(content=text))]
        return r

    def test_transient_api_error_is_retried_with_backoff_then_succeeds(self):
        c = agent.NimController(api_key='dummy')
        c.client = mock.Mock()
        c.client.chat.completions.create.side_effect = [RuntimeError('Error code: 429'), RuntimeError('Error code: 429'), self._ok('{"a": 1}')]
        with mock.patch.object(agent.time, 'sleep') as sleep:
            self.assertEqual(c._chat('p', 100), '{"a": 1}')
        self.assertEqual(sleep.call_count, 2)
        self.assertLess(sleep.call_args_list[0].args[0], sleep.call_args_list[1].args[0])

    def test_persistent_api_error_is_raised_after_limited_attempts(self):
        c = agent.NimController(api_key='dummy')
        c.client = mock.Mock()
        c.client.chat.completions.create.side_effect = RuntimeError('Error code: 429')
        with mock.patch.object(agent.time, 'sleep'):
            with self.assertRaises(RuntimeError):
                c._chat('p', 100)
        self.assertEqual(c.client.chat.completions.create.call_count, 4)

    def test_temperature_is_passed_through(self):
        c = agent.NimController(api_key='dummy')
        c.client = mock.Mock()
        c.client.chat.completions.create.return_value = self._ok('{"a": 1}')
        c._chat('p', 100, temperature=0.0)
        self.assertEqual(c.client.chat.completions.create.call_args.kwargs['temperature'], 0.0)


class AnswerPromptTests(unittest.TestCase):
    def test_answer_prompt_requires_vietnamese_and_verifier_judges_main_point(self):
        c = agent.NimController(api_key='dummy')
        sent = []
        c._chat = lambda prompt, max_tokens, retries=2, **kw: (sent.append(prompt) or
            '{"action": "answer", "answer": "Không đủ bằng chứng để trả lời.", "citations": []}')
        c.decide('q?', [{'context': '[B-1 | d | page 1]\nx', 'context_ids': ['B-1']}])
        self.assertIn('Vietnamese', sent[0])
        self.assertIn('Never answer in English', sent[0])
        sent.clear()
        c._chat = lambda prompt, max_tokens, retries=2, **kw: (sent.append(prompt) or
            '{"main_point": "m", "quote": "bằng chứng nguyên văn", "supported": true}')
        self.assertTrue(c._verify_support('q?', 'a', 'Đây là bằng chứng nguyên văn.'))
        self.assertIn('Ý CHÍNH', sent[0])


if __name__ == '__main__':
    unittest.main()
