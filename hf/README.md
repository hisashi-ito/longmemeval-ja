---
license: mit
language:
- ja
pretty_name: LongMemEval-JA
size_categories:
- n<1K
task_categories:
- question-answering
tags:
- long-term-memory
- chat-assistant
- benchmark
- japanese
configs:
- config_name: oracle
  data_files: longmemeval_ja_oracle.json
- config_name: s
  data_files: longmemeval_ja_s.json
---

<div align="center">
  <img src="https://huggingface.co/datasets/hisashi-ito/longmemeval-ja/resolve/main/assets/longmemeval-ja_logo.png" alt="LongMemEval-JA" width="75%" />
  <br/>
  <strong>English</strong> | <a href="https://huggingface.co/datasets/hisashi-ito/longmemeval-ja/blob/main/README_ja.md">日本語</a>
</div>

# LongMemEval-JA

LongMemEval-JA is an unofficial Japanese remake of [LongMemEval](https://github.com/xiaowu0162/LongMemEval) (Wu et al., ICLR 2025), a benchmark for the long-term memory of chat assistants. The conversations were not translated but rewritten for everyday life in Japan: yen instead of dollars, metric units, Japanese stores, places and holidays. The benchmark's design is unchanged: the 500 questions, the question types, the evidence sessions, the dates and the position of the evidence in the history.

The scripts, prompts and the full description of how the data was built are in the GitHub repository [hisashi-ito/longmemeval-ja](https://github.com/hisashi-ito/longmemeval-ja).

## Quick start

```python
from datasets import load_dataset

oracle = load_dataset("hisashi-ito/longmemeval-ja", "oracle", split="train")
s = load_dataset("hisashi-ito/longmemeval-ja", "s", split="train")
```

## Configurations

| Config | Questions | History per question | Size | Corresponding original |
|---|---|---|---|---|
| `oracle` | 500 | Evidence sessions only | 13 MB | `longmemeval_oracle` |
| `s` | 500 | Evidence sessions mixed into about 48 unrelated sessions (median 518 turns, about 200k Japanese characters) | 287 MB | `longmemeval_s_cleaned` |

Scores on `oracle` must not be reported as LongMemEval scores, because `oracle` does not include finding the evidence in a long history.

## Question types

| `question_type` | Questions | What it tests |
|---|---|---|
| `single-session-user` | 70 | A fact the user said in one session |
| `single-session-assistant` | 56 | A fact the assistant said in one session |
| `single-session-preference` | 30 | Using the user's preferences in a recommendation |
| `multi-session` | 133 | Combining facts across sessions |
| `knowledge-update` | 78 | Using the latest value when a fact changed |
| `temporal-reasoning` | 133 | Reasoning with dates and the order of events |

30 questions whose `question_id` ends in `_abs` are abstention questions: the history has no answer, and the expected behavior is to say so.

## Data fields

The fields and their types are the same as in the original LongMemEval, except that `answer` is always a string.

| Field | Type | Meaning |
|---|---|---|
| `question_id` | string | Same id as the original question |
| `question_type` | string | One of the six types above |
| `question` | string | The question, in Japanese |
| `answer` | string | The gold answer, in Japanese |
| `question_date` | string | When the question is asked, e.g. `2023/05/30 (Tue) 04:48` |
| `haystack_session_ids` | list of string | Ids of the sessions in the history, oldest first |
| `haystack_dates` | list of string | Date of each session |
| `haystack_sessions` | list of list of turn | The sessions; each turn has `role` (`user` or `assistant`), `content` and `has_answer` |
| `answer_session_ids` | list of string | Ids of the evidence sessions |

`has_answer` is `true` on the turns that contain the evidence.

## Example

From `oracle`, question `66f24dbb` (single-session-user):

```json
{
  "question_id": "66f24dbb",
  "question_type": "single-session-user",
  "question": "妹の誕生日プレゼントには何を買ったっけ?",
  "answer": "黄色いワンピース",
  "question_date": "2023/05/30 (Tue) 04:48",
  "haystack_session_ids": ["answer_fea2e4d3"],
  "haystack_dates": ["2023/05/26 (Fri) 11:54"],
  "answer_session_ids": ["answer_fea2e4d3"]
}
```

The evidence turn in that session (`has_answer: true`):

```json
{"role": "user", "content": "妹の誕生日には、黄色いワンピースと、それに合わせたイヤリングをあげたよ。", "has_answer": true}
```

## How it was made

| Part | Count | Source | How it was made |
|---|---|---|---|
| Questions, answers and evidence sessions | 500 questions, 940 sessions | LongMemEval (MIT) | Rewritten with Claude Sonnet 5.5 |
| Unrelated sessions written by the LongMemEval authors | 8,557 | LongMemEval (MIT) | Rewritten with Claude Haiku 5.5 |
| Unrelated sessions from UltraChat | 5,173 | UltraChat (MIT) | Rewritten with Claude Haiku 5.5 |
| Unrelated sessions in the positions of ShareGPT conversations | 4,525 | None | Newly written in Japanese with Claude Haiku 5.5 |

ShareGPT conversations are real users' conversations collected without the writers' own permission, so they are neither translated nor included. A model read each one and wrote only a one-line, generalized topic; the new conversation was written from that line alone. Every new conversation was screened for harmful content, and 20 were rewritten from other topics.

## Quality checks

| Check | Result |
|---|---|
| Shape | Number and order of sessions and turns, roles, positions of `has_answer` and all dates match the original |
| Answers | Claude Opus 5.5 answered from the rewritten sessions alone. On `oracle`, 474 of 500 are correct, against 475 for the English original solved the same way |
| Leakage | For short answers, the answer string appears in an unrelated session for 16% of questions, against 28% in the English original |

<img src="https://huggingface.co/datasets/hisashi-ito/longmemeval-ja/resolve/main/figures/en/verification.png" alt="Accuracy per question type" width="80%" />

More figures (history length, evidence position, sources, length ratio) and the details of each check are in the [GitHub repository](https://github.com/hisashi-ito/longmemeval-ja).

## Limitations

- Do not compare Japanese scores directly with scores on the English original; the content and length of the sessions differ
- The Japanese sessions are shorter than the English ones (median character ratio 0.35); long replies were shortened, but the evidence and the numbers needed for the answers were kept
- The new sessions do not have the same number of turns as the ShareGPT sessions they replace
- As in the original, 13 questions list the same session id twice, and some sessions are cut off
- All Japanese text was generated by Claude; names of real Japanese companies and products appear in everyday contexts, as US ones do in the original

## License

MIT. Copyright (c) 2026 Hisashi Itoh. LongMemEval and UltraChat are also MIT; their notices are in [`LICENSE.upstream`](https://huggingface.co/datasets/hisashi-ito/longmemeval-ja/blob/main/LICENSE.upstream) and [`NOTICE.md`](https://huggingface.co/datasets/hisashi-ito/longmemeval-ja/blob/main/NOTICE.md).

## Citation

This is not an official release by the LongMemEval authors. If you use this dataset, please cite the original works:

```bibtex
@inproceedings{wu2025longmemeval,
  title     = {LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory},
  author    = {Di Wu and Hongwei Wang and Wenhao Yu and Yuwei Zhang and Kai-Wei Chang and Dong Yu},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2025},
  note      = {arXiv:2410.10813}
}

@article{ding2023ultrachat,
  title   = {Enhancing Chat Language Models by Scaling High-quality Instructional Conversations},
  author  = {Ning Ding and Yulin Chen and Bokai Xu and Yujia Qin and Zhi Zheng and Shengding Hu and
             Zhiyuan Liu and Maosong Sun and Bowen Zhou},
  journal = {arXiv preprint arXiv:2305.14233},
  year    = {2023}
}
```
