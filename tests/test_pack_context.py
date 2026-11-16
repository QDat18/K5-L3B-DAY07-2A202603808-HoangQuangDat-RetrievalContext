import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'starter'))
from baseline import pack_context


def chunk(id, text, page=1):
    return {'id': id, 'source_id': 'doc', 'source': 'd.docx', 'page': page, 'text': text}


class PackContextTests(unittest.TestCase):
    def test_everything_fits_is_unchanged(self):
        cs = [chunk('a', 'Một.'), chunk('b', 'Hai.')]
        context, ids = pack_context(cs, 800)
        self.assertEqual(ids, ['a', 'b'])
        self.assertEqual(context, '[a | d.docx | page 1]\nMột.\n\n[b | d.docx | page 1]\nHai.')

    def test_overflow_keeps_every_chunk_by_truncating_instead_of_dropping(self):
        # q03: 3 chunk ~700 ký tự, budget 1500 -> trước đây chỉ còn 1 chunk
        cs = [chunk(i, ('từ ' * 400)[:700]) for i in 'abc']
        context, ids = pack_context(cs, 1500)
        self.assertEqual(ids, ['a', 'b', 'c'])
        self.assertLessEqual(len(context), 1500)
        for i in 'abc':
            self.assertIn(f'[{i} | d.docx | page 1]', context)
        self.assertIn('…', context)

    def test_top_ranked_chunk_gets_priority_over_lower_ranked(self):
        # q13: bằng chứng nằm cuối chunk hạng 1 (755 ký tự), chia đều 1/3 sẽ cắt mất
        cs = [chunk('a', 'a' * 755), chunk('b', 'b' * 716), chunk('c', 'c' * 685)]
        context, ids = pack_context(cs, 1500)
        self.assertEqual(ids, ['a', 'b', 'c'])
        self.assertIn('a' * 755, context)
        self.assertLessEqual(len(context), 1500)

    def test_short_chunk_is_not_truncated_when_others_are_long(self):
        cs = [chunk('a', 'x' * 900), chunk('b', 'Ngắn gọn.'), chunk('c', 'y' * 900)]
        context, ids = pack_context(cs, 1000)
        self.assertEqual(ids, ['a', 'b', 'c'])
        self.assertIn('\nNgắn gọn.', context)
        self.assertLessEqual(len(context), 1000)

    def test_chunk_whose_share_is_too_small_is_dropped_not_kept_as_stub(self):
        cs = [chunk(i, 'z' * 700) for i in 'abcdef']
        context, ids = pack_context(cs, 700)
        self.assertLessEqual(len(context), 700)
        self.assertTrue(ids and len(ids) < 6)
        self.assertEqual(ids, list('abcdef')[:len(ids)])  # giữ thứ hạng cao nhất

    def test_tiny_budget_and_dedup(self):
        self.assertEqual(pack_context([chunk('a', 'Một.')], 1), ('', []))
        _, ids = pack_context([chunk('a', 'Một.'), chunk('a', 'Một.')], 800)
        self.assertEqual(ids, ['a'])


if __name__ == '__main__':
    unittest.main()
