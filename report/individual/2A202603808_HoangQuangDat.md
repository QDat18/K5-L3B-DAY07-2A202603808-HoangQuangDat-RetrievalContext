# Báo cáo Day 7 — Hoàng Quang Đạt (2A202603808)

## Dữ liệu và benchmark

- **Nguồn, quyền sử dụng, số trang, số chunk:** Quyết định 5481/QĐ-BYT của Bộ Y tế, hướng dẫn chẩn đoán và điều trị đái tháo đường típ 2. Đây là văn bản quy phạm kỹ thuật/hướng dẫn chuyên môn của Bộ Y tế ban hành công khai, thuộc diện văn bản nhà nước phi thương mại được phép sử dụng cho mục đích nghiên cứu, học tập theo Luật Sở hữu trí tuệ Việt Nam. Có 80 trang, cắt thành 234 chunk (trung bình 585,2 ký tự, ngắn nhất 45, dài nhất 800).
- **Vector store, embedding:** PostgreSQL + pgvector (làm từ Day 6). Embedding `BAAI/bge-m3`, 1024 chiều. Reranker `BAAI/bge-reranker-base`. Phiên bản thư viện: `sentence-transformers 6.3.0`, `transformers 5.22.2`, `torch 2.15.0+cpu`, `psycopg 3.3.8`, `openai 3.51.0`.
- **Agent model, cách chạy, chia dev/test:** Model `nvidia/nemotron-3.5-lightning-30b-a3b` gọi qua NVIDIA NIM, tắt chế độ "thinking", temperature 0,1 (bước kiểm chứng dùng 0). Dev 17 câu (15 có đáp án + 2 thiếu bằng chứng), test 7 câu (5 + 2). Dev dùng để chỉnh. Test chạy một lần, không sửa gì sau khi xem.
- **Cách xác định có/thiếu evidence, tiêu chí chấm:** Câu có đáp án ghi sẵn nguồn, trang và đoạn evidence nằm trọn trong một chunk. Câu thiếu bằng chứng là câu gần chủ đề nhưng nội dung không có trong văn bản, mỗi câu có ghi lý do trong `data/questions.json`. Một chunk "trúng" khi đúng nguồn, đúng trang và chứa nguyên evidence.

## Retrieval và thử nghiệm

Số retrieval của agent lấy từ trace (top-3, mỗi câu 1 lần tìm). Số "baseline TF-IDF" lấy từ `starter/results/dev-tfidf/summary.json` và `test-tfidf/summary.json`.

| Cấu hình | Phương pháp, top-k, rerank, budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|---|---|---|
| Baseline | TF-IDF, top-3, budget 800 | Mốc so sánh có sẵn | dev | 10/15 (66,7%) | 12/15 (80,0%) | 6/15 (40,0%) | 1,1 ms/câu, 0 USD |
| Phương pháp 2 | Dense (pgvector + bge-m3), top-3 | Hiểu nghĩa tốt hơn từ khóa | dev | 9/15 (60,0%) | 13/15 (86,7%) | 13/15 (86,7%) | 212,7 ms/câu, 0 USD |
| Phương pháp 3 | Lexical BM25, top-3 | Hợp với thuật ngữ y khoa | dev | 8/15 (53,3%) | 15/15 (100,0%) | 10/15 (66,7%) | 3,1 ms/câu, 0 USD |
| Rerank, lần đầu | Dense lấy 10 → rerank còn 3, budget 1500; query do model viết lại | Rerank sẽ xếp đúng hơn | dev | 1/15 (6,7%) | 4/15 (26,7%) | 2/15 (13,3%) | 1670,3 ms/câu, 0 USD |
| Cải thiện 1 | Như trên nhưng **query = nguyên câu hỏi** | Query model viết lại làm lệch kết quả | dev | 12/15 (80,0%) | 13/15 (86,7%) | 12/15 (80,0%) | 1684,2 ms/câu, 0 USD |
| Cải thiện 2 | Thêm **cắt bớt chunk dài thay vì bỏ chunk** khi đóng gói context | Chunk bằng chứng bị bỏ do hết chỗ | dev | 12/15 (80,0%) | 13/15 (86,7%) | 13/15 (86,7%) | 1658,4 ms/câu, 0 USD |
| Cấu hình đã chọn | Rerank + 2 cải thiện trên | — | test | 3/5 (60,0%) | 3/5 (60,0%) | 3/5 (60,0%) | 1744,5 ms/câu, 0 USD |
| Dense (đối chiếu) | Dense, top-3, budget 1500 | — | test | 3/5 (60,0%) | 3/5 (60,0%) | 3/5 (60,0%) | 197,3 ms/câu, 0 USD |
| Lexical (đối chiếu) | BM25, top-3, budget 1500 | — | test | 4/5 (80,0%) | 5/5 (100,0%) | 5/5 (100,0%) | 1,8 ms/câu, 0 USD |
| Baseline | TF-IDF, top-3, budget 800 | Mốc đối chiếu test | test | 4/5 (80,0%) | 4/5 (80,0%) | 4/5 (80,0%) | 0,92 ms/câu, 0 USD |

**Ba lỗi chính trên dev, nguyên nhân và cách sửa:**

1. **Query biến thành chữ mẫu.** Model "suy nghĩ" quá lâu nên hết token, phần trả lời trống. Code lại lấy JSON nằm trong phần suy nghĩ, nên query thành `"search keywords or search phrase"` và nhiều câu trả lời là chữ mẫu. Sửa: tắt suy nghĩ, chỉ dùng phần trả lời, nếu trống hoặc bị cắt thì gọi lại. Sau sửa câu bị abstain vì lý do này không còn.
2. **Query do model viết lại làm lệch retrieval.** Model viết query tiếng Anh kèm mã "5481/QĐ-BYT", trong khi tài liệu là tiếng Việt. Hit@1 chỉ 1/15 (6,7%). Sửa: dùng nguyên câu hỏi tiếng Việt, Hit@1 lên 12/15 (80%) và không cần gọi model thêm một lượt.
3. **Chunk bằng chứng bị bỏ vì hết chỗ.** Hàm đóng gói context bỏ cả chunk không vừa 1500 ký tự. Câu q03 có chunk đúng ở hạng 3 nhưng bị loại. Sửa: cắt phần đuôi chunk dài, chunk hạng 1 được ưu tiên tối đa 60% chỗ. Đã kiểm lại offline: câu nào trước đó giữ được evidence vẫn giữ, thêm q03.

Kèm hai lỗi nhỏ: model trả `citations` có dấu ngoặc `[B-...]` nên bị coi là không hợp lệ (làm mất q03, q06) và gọi API bị lỗi 429 làm abstain oan. Cả hai đã sửa.

**Vì sao chọn cấu hình này:** rerank + query gốc + cắt chunk cho Hit@1 và Context Rate cao nhất trên dev, kèm các lớp chống bịa ở phần dưới. Trên test (5 câu) chưa thấy rerank hơn hẳn dense và lexical, xem phần giới hạn.

## Agent và đánh giá

- **Tool, số lượt gọi, điều kiện dừng:** 1 tool `search` (retrieval). Mỗi câu agent tìm 1 lần (17/17 câu dev và 7/7 câu test), tối đa cho phép 3 lần. Dừng khi model trả lời có trích dẫn hợp lệ, hoặc khi từ chối.
- **Cách chọn context:** lấy top-3 sau rerank, xếp vào 1500 ký tự; chunk dài bị cắt đuôi, chunk bị cắt có dấu "…".
- **Chống bịa, theo thứ tự:** (1) model trả lời bằng tiếng Việt, chỉ dùng đoạn trích, kết luận nêu trước; (2) citation phải là chunk có trong context; (3) số trong câu trả lời phải có trong chunk được trích (không tính số lặp lại từ câu hỏi, số thứ tự "1)", ID chunk); (4) một lượt model nữa kiểm xem ý chính có được chunk nói rõ không, bắt buộc kèm câu trích nguyên văn và code kiểm câu đó có thật trong chunk. Câu nào không qua thì trả "Không đủ bằng chứng để trả lời."
- **Xử lý lỗi:** trả lời trống hoặc bị cắt thì gọi lại với giới hạn token gấp đôi; lỗi API tạm thời (429, mạng) thì chờ 1-2-4 giây rồi thử lại; bước kiểm chứng bị lỗi thì coi như không được hỗ trợ (từ chối, không đoán bừa).

| Cấu hình / split | Tool use | Số lượt gọi | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|
| Rerank, dev (chốt v6) | 17/17 | 1 lượt/câu | 1 / 2 / 12 câu; TB 1,73/2 (26/30) | 13/21 trỏ đúng chunk chứa evidence* | 15/15 | 2/2 | 0/15 |
| Rerank, test (chọn) | 7/7 | 1 lượt/câu | 2 / 0 / 3 câu; TB 1,2/2 (6/10) | 3/4 | 4/4 | 2/2 | 1/5 |
| Dense, test | 7/7 | 1 lượt/câu | 2 / 0 / 3 câu; TB 1,2/2 (6/10) | 3/3 | 3/3 | 2/2 | 2/5 |
| Lexical, test | 7/7 | 1 lượt/câu | 2 / 0 / 3 câu; TB 1,2/2 (6/10) | 3/3 | 3/3 | 2/2 | 2/5 |

\* Đối soát thủ công 5 câu test: context đạt mức 2 (đủ evidence và nguồn) ở 3 câu (q08, q11, q14); đạt mức 0 ở 2 câu (q18, q20 do không lấy trúng chunk gold vào top-3 context). Điểm context trung bình trên test: 1,2/2 (6/10). Về citation: 3/4 citation trỏ đúng chunk gold hỗ trợ trực tiếp câu trả lời (q08, q11, q14), citation ở q20 trỏ sai nguồn hỗ trợ. Trên dev, 8 citation còn lại là chunk phụ (như q03, q05, q07, q09, q19) bổ sung bối cảnh điều trị nhưng không chứa trọn vẹn chuỗi text evidence gold.

**Chi tiết từng câu test (cấu hình chọn: rerank).** Tool call là 1 lần `search` bằng nguyên câu hỏi, top-3.

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| q08 | Có đáp án | search → U055-003, U056-001, U056-003 (có evidence ở hạng 1) | Insulin nền (NPH hoặc detemir), liều 0,05–0,1 IU/kg/ngày hoặc 2–3 UI, tăng 1–2 IU mỗi 2–3 ngày. [U055-003] | **2**. Đúng và đủ, citation đúng chunk |
| q11 | Có đáp án | search → U014-001, U011-003, U019-001 | HbA1c <8,0%, glucose đói/trước ăn 90–150 mg/dL (thêm 100–180 lúc đi ngủ). [U014-001] | **2**. Đúng, phần thêm vẫn nằm trong chunk |
| q14 | Có đáp án | search → U004-002, U002-001, U004-001 | Chẩn đoán khi glucose huyết tương lúc đói ≥ 92 mg/dL (5,1 mmol/L). [U004-002] | **2**. Đúng |
| q18 | Có đáp án | search → U071-003, U065-002, U072-001 (không có evidence) | Từ chối: "Không đủ bằng chứng". | **0**. Từ chối nhầm, vì chunk liraglutide không được tìm ra. Từ chối là hợp lý khi không có bằng chứng |
| q20 | Có đáp án | search → U077-001, U076-003, U074-002 (không có evidence) | "Cách nhau tối thiểu 42 giờ". [U077-001] | **0**. **Sai**: đáp án đúng là 8 giờ. Model chép số 42 từ câu hỏi; chunk gold không được tìm ra |
| q_unans_test_01 | Thiếu bằng chứng (ADA 2024) | search → U004-001, U004-002, U054-001 | Từ chối | Đúng. Model từng soạn câu trả lời từ chunk nên bị bước kiểm chứng chặn, không bịa thêm |
| q_unans_test_02 | Thiếu bằng chứng (Teplizumab) | search → U032-001, U031-001, U047-002 | Từ chối | Đúng |

- **Đường dẫn trace, lệnh chạy:** dev `starter/results/agent_dev_v6/traces.json` (lệnh: `python src/agent.py --split dev --method rerank --budget 1500 --output starter/results/agent_dev_v6`); test `starter/results/agent_test_{rerank,dense,lexical}/traces.json` (cùng lệnh, đổi `--split test --method ...`). Các lần chạy dev cũ `agent_dev`, `agent_dev_v2`…`v5` giữ lại để so trước/sau. Thời gian chạy đo đạc thực tế: trung bình 18,7 giây/câu (dao động từ 17,6 s/câu trên dev đến 21,4 s/câu trên test, gồm 1 lượt retrieval, 1 lượt model sinh câu trả lời và 1 lượt verifier); chi phí: 0 USD (dùng free credits của NVIDIA NIM, model embedding và reranker chạy local trên CPU).
- **Kết quả trước/sau (dev, câu có đáp án, tổng 15):**

| Lần chạy | Câu được trả lời | Câu đúng gold | Ghi chú |
|---|---|---|---|
| v1 (ban đầu) | 5 | 3 (q03, q04, q07) | query chữ mẫu, nhiều câu tiếng Anh, có câu bịa liều (q19) và câu trả lời rỗng nội dung (q01) |
| v3 (query gốc) | 9 | 8 | q12 trả lời lệch phạm vi (về ĐTĐ típ 1 thay vì mốc glucose an toàn) |
| v6 (chốt) | 15 | 13 | q06 sai (gán lối sống thay vì insulin), q12 và q19 đúng một phần |

- **Ví dụ lỗi và đánh đổi:**
  - Các lớp chống bịa giúp chặn nhiều câu sai, nhưng không chặn được khi tìm sai chunk mà câu trả lời vẫn có căn cứ với chunk đó. q06 trả lời "không cần can thiệp thuốc" trong khi đáp án đúng là cân nhắc insulin sớm (HbA1c ≥9% hoặc sụt cân). Đo đạc trace cho thấy retriever trả về U012-002 và U030-003 (nói về thay đổi lối sống) nên model đã trả lời hoàn toàn dựa vào 2 chunk này dù sai lệch so với đáp án y khoa chính thức.
  - Trên test, kiểm tra "số lặp lại từ câu hỏi thì bỏ qua" để câu như "mức 2" không bị chặn oan lại tạo lỗ hổng: q20 lọt vì "42 giờ" nằm sẵn trong câu hỏi và xuất hiện trong chunk U077-001 (thời gian tác dụng của degludec), dẫn đến model trả lời sai "cách nhau tối thiểu 42 giờ" thay vì "8 giờ". Với Dense và Lexical trên q20, do không truy xuất được chunk chứa mốc 8 giờ (U077-002) vào top-3 và thiếu bằng chứng trích dẫn, hệ thống kích hoạt sự kiện `invalid_or_missing_citations` và chủ động từ chối (abstain).
  - Lexical trên q18 tìm được chunk gold U067-001 ở vị trí thứ 2, nhưng khi đóng gói context 1500 ký tự, chunk này bị cắt cụt đuôi ngay tại `sau 1…`, làm mất thông tin liều tối đa 1,8 mg/ngày. Do thiếu bằng chứng đầy đủ trong context bị cắt, agent không thể đưa ra trích dẫn hợp lệ cho toàn bộ câu hỏi và kích hoạt cơ chế an toàn từ chối (abstain).
- **Giới hạn:** test chỉ có 5 câu có đáp án nên 1 câu lệch là 20%, chưa kết luận được rerank hơn dense hay lexical (lexical còn có Hit@3 cao nhất, 5/5). Dev là bộ đã dùng để chỉnh nên số dev lạc quan hơn thật. Mô hình còn dao động giữa các lần chạy (q06, q16, q19 lúc trả lời lúc từ chối khi mô phỏng lại).  