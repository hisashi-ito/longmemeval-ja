<div align="center">
  <img src="https://huggingface.co/datasets/hisashi-ito/longmemeval-ja/resolve/main/assets/longmemeval-ja_logo.png" alt="LongMemEval-JA" width="75%" />
  <br/>
  <a href="https://huggingface.co/datasets/hisashi-ito/longmemeval-ja">English</a> | <strong>日本語</strong>
</div>

# LongMemEval-JA

LongMemEval-JA は、チャットアシスタントの長期記憶を測るベンチマーク [LongMemEval](https://github.com/xiaowu0162/LongMemEval)(Wu et al., ICLR 2025)を、日本語で作り直した非公式の版です。会話は訳さずに、日本の暮らしに合わせて書き直しました。ドルは円に、マイルはキロメートルに、米国の店や祝日は日本のものにしています。500 問の質問、問題の種類、根拠の会話、日付、履歴の中での根拠の位置といった、ベンチマークの作りは元のままです。

データを作ったスクリプト、モデルへの指示文、作り方の詳しい説明は、GitHub のリポジトリ [hisashi-ito/longmemeval-ja](https://github.com/hisashi-ito/longmemeval-ja) にあります。

## 読み込み方

```python
from datasets import load_dataset

oracle = load_dataset("hisashi-ito/longmemeval-ja", "oracle", split="train")
s = load_dataset("hisashi-ito/longmemeval-ja", "s", split="train")
```

## 版

| 版 | 問題数 | 1 問あたりの履歴 | 大きさ | 対応する元の版 |
|---|---|---|---|---|
| `oracle` | 500 | 根拠の会話だけ | 13MB | `longmemeval_oracle` |
| `s` | 500 | 根拠の会話を、約 48 本の関係ない会話に混ぜたもの。発言は中央値で 518 個、日本語で約 20 万字 | 287MB | `longmemeval_s_cleaned` |

`oracle` には、長い履歴の中から根拠を探す部分がありません。そのため、`oracle` の点数を LongMemEval の点数として報告しないでください。

## 問題の種類

| `question_type` | 問題数 | 測ること |
|---|---|---|
| `single-session-user` | 70 | 1 つの会話の中で、利用者が話した事実 |
| `single-session-assistant` | 56 | 1 つの会話の中で、アシスタントが話した事実 |
| `single-session-preference` | 30 | 利用者の好みを踏まえた提案 |
| `multi-session` | 133 | 複数の会話にまたがる事実の組み合わせ |
| `knowledge-update` | 78 | 事実が変わったときに、新しいほうを使えるか |
| `temporal-reasoning` | 133 | 日付と出来事の前後関係を使った推論 |

`question_id` が `_abs` で終わる 30 問は、答えない判断の問題です。履歴に答えが無いので、答えが無いと言えるかを見ます。

## 項目

項目と型は元の LongMemEval と同じです。ただし `answer` は、すべて文字列にそろえました。

| 項目 | 型 | 意味 |
|---|---|---|
| `question_id` | 文字列 | 元の問題と同じ ID |
| `question_type` | 文字列 | 上の 6 種類のどれか |
| `question` | 文字列 | 日本語の質問 |
| `answer` | 文字列 | 日本語の正解 |
| `question_date` | 文字列 | 質問した日時。例: `2023/05/30 (Tue) 04:48` |
| `haystack_session_ids` | 文字列のリスト | 履歴の会話の ID。古い順 |
| `haystack_dates` | 文字列のリスト | 各会話の日時 |
| `haystack_sessions` | 発言のリストのリスト | 会話の本体。各発言は `role`(`user` か `assistant`)、`content`、`has_answer` を持つ |
| `answer_session_ids` | 文字列のリスト | 根拠の会話の ID |

`has_answer` は、根拠を含む発言で `true` になります。

## 例

`oracle` の問題 `66f24dbb`(single-session-user)です。

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

この会話の中の、根拠の発言(`has_answer: true`)はこれです。

```json
{"role": "user", "content": "妹の誕生日には、黄色いワンピースと、それに合わせたイヤリングをあげたよ。", "has_answer": true}
```

## 作り方

| 部分 | 数 | 出どころ | 作り方 |
|---|---|---|---|
| 質問、答え、根拠の会話 | 500 問、940 本 | LongMemEval(MIT) | Claude Sonnet 5.5 で作り直した |
| LongMemEval の著者が書いた関係ない会話 | 8,557 本 | LongMemEval(MIT) | Claude Haiku 5.5 で作り直した |
| UltraChat の関係ない会話 | 5,173 本 | UltraChat(MIT) | Claude Haiku 5.5 で作り直した |
| ShareGPT の会話があった位置の関係ない会話 | 4,525 本 | 無し | Claude Haiku 5.5 で日本語で新しく書いた |

ShareGPT の会話は、実在の利用者の会話を、書いた本人の許諾なしに集めたものです。そのため訳さず、データにも入れていません。モデルに ShareGPT の会話を読ませて、一般化した 1 行の話題だけを書き出させ、新しい会話はその 1 行だけを見て書かせました。新しい会話は、すべて有害な中身がないかを判定し、20 本を別の話題で書き直しています。

## 品質の検査

| 検査 | 結果 |
|---|---|
| 構造 | 会話と発言の数と順番、話し手、`has_answer` の位置、すべての日付が、元と同じ |
| 答え | Claude Opus 5.5 に、作り直した会話だけを読ませて答えさせた。`oracle` で 500 問中 474 問が正解。同じ方法で解かせた英語の元の版は 475 問 |
| 漏れ | 答えが短い問題で、答えの文字列が関係ない会話に出てくるのは 16%。英語の元の版は 28% |

<img src="https://huggingface.co/datasets/hisashi-ito/longmemeval-ja/resolve/main/figures/ja/verification.png" alt="問題の種類ごとの正解の割合" width="80%" />

ほかの図(履歴の長さ、根拠の位置、出どころ、字数の比)と、検査の詳しい中身は、[GitHub のリポジトリ](https://github.com/hisashi-ito/longmemeval-ja)にあります。

## 使うときの注意

- 日本語版の点数を、英語の元の版の点数とそのまま比べないでください。会話の中身も長さも違います
- 日本語の会話は、英語より短くなっています。字数の比は中央値で 0.35 です。長い返事は縮めましたが、根拠と、答えに要る数は残しています
- 新しく書いた会話は、置き換えた ShareGPT の会話と、発言の数がそろっていません
- 元の版と同じく、同じ会話の ID が 2 度出てくる問題が 13 問あり、途中で切れている会話もあります
- 日本語の文は、すべて Claude が生成したものです。元の版に米国の会社や商品が出てくるのと同じように、日本の実在の会社や商品の名前が、暮らしの話の中に出てきます

## ライセンス

MIT ライセンスです。Copyright (c) 2026 Hisashi Itoh。LongMemEval と UltraChat も MIT ライセンスで、その表示は [`LICENSE.upstream`](https://huggingface.co/datasets/hisashi-ito/longmemeval-ja/blob/main/LICENSE.upstream) と [`NOTICE.md`](https://huggingface.co/datasets/hisashi-ito/longmemeval-ja/blob/main/NOTICE.md) にあります。

## 引用

LongMemEval の著者による公式の版ではありません。このデータを使うときは、元の論文を引用してください。BibTeX は[英語のカード](https://huggingface.co/datasets/hisashi-ito/longmemeval-ja)にあります。

- Di Wu, Hongwei Wang, Wenhao Yu, Yuwei Zhang, Kai-Wei Chang, Dong Yu. LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory. ICLR 2025. arXiv:2410.10813
- Ning Ding, Yulin Chen, Bokai Xu, Yujia Qin, Zhi Zheng, Shengding Hu, Zhiyuan Liu, Maosong Sun, Bowen Zhou. Enhancing Chat Language Models by Scaling High-quality Instructional Conversations. arXiv:2305.14233, 2023
