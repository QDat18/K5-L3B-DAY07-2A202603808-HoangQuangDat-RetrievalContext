import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from grounding import split_context, ungrounded_numbers, cited_text, quote_in_evidence

CTX = ("[B-1 | d.docx | page 1]\nLDL <100 mg/dL (2,6 mmol/L); HbA1c <7,0%.\n\n"
       "[B-2 | d.docx | page 2]\nSitagliptin: 50 mg/ngày khi eGFR 30-50.")


class GroundingTests(unittest.TestCase):
    def test_split_context_maps_id_to_text(self):
        m = split_context(CTX)
        self.assertEqual(set(m), {'B-1', 'B-2'})
        self.assertIn('HbA1c <7,0%', m['B-1'])
        self.assertNotIn('Sitagliptin', m['B-1'])

    def test_cited_text_only_joins_cited_chunks(self):
        t = cited_text(CTX, ['B-2'])
        self.assertIn('Sitagliptin', t)
        self.assertNotIn('LDL', t)

    def test_number_not_in_cited_chunk_is_flagged(self):
        # q02: số 54 bịa, trích chunk lipid
        ans = 'Hạ đường huyết mức 2 khi dưới 54 mg/dL (3,0 mmol/L). [B-1]'
        q = 'Hạ đường huyết mức 2 được xác định khi glucose huyết ở ngưỡng nào?'
        self.assertEqual(ungrounded_numbers(ans, cited_text(CTX, ['B-1']), q), ['54', '3'])

    def test_numbers_restated_from_question_are_allowed(self):
        self.assertEqual(ungrounded_numbers('eGFR 40 thì giảm liều [B-2]', 'giảm liều', 'eGFR 40 xử trí?'), [])

    def test_decimal_comma_and_dot_are_equivalent(self):
        ans = 'Mục tiêu HbA1c < 7.0% [B-1]'
        self.assertEqual(ungrounded_numbers(ans, cited_text(CTX, ['B-1'])), [])

    def test_citation_ids_are_not_treated_as_numbers(self):
        ans = 'HbA1c <7,0% [B-5481-U013-002]'
        self.assertEqual(ungrounded_numbers(ans, 'HbA1c <7,0%'), [])

    def test_multiple_ids_in_one_bracket_or_bare_are_ignored(self):
        # q_unans_dev_02: model xuất ID dạng [id1, id2] -> trước đây bị coi là số 5481, 018, 002...
        ans = 'Không có thông tin [B-5481-U018-002, B-5481-U019-003] và B-5481-U019-003'
        self.assertEqual(ungrounded_numbers(ans, 'x'), [])

    def test_list_enumerators_are_not_numbers(self):
        # q03/q06: "1) ... 2) ..." và "3. ..." đầu dòng bị coi là số lạ
        ans = '1) Cho uống 15 g glucose.\n2) Thử lại sau 15 phút. 3) Ăn nhẹ.\n4. Theo dõi'
        self.assertEqual(ungrounded_numbers(ans, '15 g glucose, 15 phút'), [])

    def test_real_number_after_enumerator_still_flagged(self):
        self.assertEqual(ungrounded_numbers('1) Uống 20 g glucose', '15 g glucose'), ['20'])

    def test_trailing_zero_decimal_equals_integer(self):
        self.assertEqual(ungrounded_numbers('HbA1c < 7%', 'HbA1c < 7,0%'), [])

    def test_quote_must_appear_in_evidence(self):
        ev = 'Những BN điều trị insulin:  nên đo đường huyết 6 lần/ngày vào trước và sau ăn 3 bữa chính.'
        self.assertTrue(quote_in_evidence('nên đo đường huyết 6 lần/ngày', ev))
        self.assertTrue(quote_in_evidence('Nên ĐO đường huyết 6 lần/ngày vào trước và sau ăn…', ev))  # khác hoa/thường, dấu …
        self.assertFalse(quote_in_evidence('nên đo đường huyết 8 lần/ngày', ev))
        self.assertFalse(quote_in_evidence('', ev))
        self.assertFalse(quote_in_evidence('đo', ev))  # quá ngắn, không đáng tin

    def test_quote_with_ellipsis_joins_separate_passages(self):
        # q19: verifier trích 2 đoạn cách nhau bằng "..." (đúng cách trích) -> từng đoạn phải có trong evidence
        ev = 'Chống chỉ định: eGFR < 30 mL/phút. Đoạn giữa không liên quan. Nên cân nhắc cẩn thận việc dùng metformin.'
        self.assertTrue(quote_in_evidence('Chống chỉ định: eGFR < 30 mL/phút... Nên cân nhắc cẩn thận việc dùng metformin.', ev))
        self.assertTrue(quote_in_evidence('Chống chỉ định: eGFR < 30 mL/phút. … Nên cân nhắc cẩn thận việc dùng metformin.', ev))

    def test_quote_with_ellipsis_rejected_if_any_passage_is_fabricated(self):
        ev = 'Chống chỉ định: eGFR < 30 mL/phút. Đoạn giữa không liên quan.'
        self.assertFalse(quote_in_evidence('Chống chỉ định: eGFR < 30 mL/phút... Nên dùng liều 50 mg mỗi ngày.', ev))

    def test_no_numbers_means_nothing_flagged(self):
        self.assertEqual(ungrounded_numbers('Nên ăn nhẹ [B-1]', 'ăn nhẹ'), [])


if __name__ == '__main__':
    unittest.main()
