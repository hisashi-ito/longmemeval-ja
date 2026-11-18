# Notice

This project is derived from LongMemEval and UltraChat. Our own scripts and data are released under the
MIT License (Copyright (c) 2026 Hisashi Itoh, full text in `LICENSE`).

## LongMemEval

- Paper: LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory. Di Wu, Hongwei Wang,
  Wenhao Yu, Yuwei Zhang, Kai-Wei Chang, Dong Yu. ICLR 2025. arXiv:2410.10813
- Code: https://github.com/xiaowu0162/LongMemEval
- Data: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned
- License: MIT License, Copyright (c) 2024 Di Wu (full text in `LICENSE.upstream`)

## UltraChat

- Paper: Enhancing Chat Language Models by Scaling High-quality Instructional Conversations. Ning Ding,
  Yulin Chen, Bokai Xu, Yujia Qin, Zhi Zheng, Shengding Hu, Zhiyuan Liu, Maosong Sun, Bowen Zhou.
  arXiv:2305.14233, 2023
- Data: https://huggingface.co/datasets/stingning/ultrachat
- License: MIT License (as declared on the dataset page)

UltraChat sessions reach this project through LongMemEval, which uses them as unrelated sessions in its
histories.

## Changes from the originals

- Questions, answers, evidence sessions and unrelated sessions are rewritten in Japanese and adapted to
  everyday life in Japan (currency, units, stores, places, holidays, institutions). Creative works keep
  their titles. Dates and the order of events are unchanged; answers that depend on changed numbers are
  recomputed.
- Unrelated sessions that LongMemEval took from ShareGPT are not translated or included. New Japanese
  sessions are written in their positions from one-line generalized topics; no ShareGPT text is included.
- The rewriting is done with large language models (Claude, through the Anthropic API), followed by
  automatic checks and human review.

This project is not affiliated with or endorsed by the authors of LongMemEval or UltraChat.
