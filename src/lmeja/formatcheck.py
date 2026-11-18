"""公開するファイルの形式の検査。

1 本ずつの検査(発言の型、空、英語の残り、作り直しの途中の言葉、コードの囲み、極端な縮み)と、組み上げた
ファイルの検査(元の LongMemEval と同じ項目と型、会話の番号と日付と本数の対応、根拠の会話の番号)を行う。
"""
import json
import logging
import re
import unicodedata

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s -- : %(message)s')
logger = logging.getLogger(__name__)

# 元の LongMemEval の 1 問の項目と型
FIELDS = {
    "question_id": str,
    "question_type": str,
    "question": str,
    "answer": str,
    "question_date": str,
    "haystack_session_ids": list,
    "haystack_dates": list,
    "haystack_sessions": list,
    "answer_session_ids": list,
}
# 作り直しの跡として、はっきりしたものだけを拾う(「置き換える」は会話の中身でも普通に出るので拾わない)
META_WORDS = re.compile(r"訳注|原文では|元の英語|ではなく…|作り直しました|に置き換えました")
# 文の終わりの印。最後の発言がこれ以外で終われば、途中で切れた疑い
ENDINGS = re.compile(r"[。！？!?」』）)\]】〕…ー～〜♪w笑草\w*`|]\s*$")
FENCE = re.compile(r"^\s*```(json)?\s*$", re.M)
LATIN = re.compile(r"[A-Za-z]")
JAPANESE = re.compile(r"[ぁ-んァ-ヶ一-龠]")


class TurnChecker:
    def __init__(self, max_latin=0.6):
        # 発言のうちアルファベットの割合がこれより大きく、日本語が無ければ、英語のまま残った疑い
        self.max_latin = max_latin

    def turns(self, turns):
        problems = []
        if not isinstance(turns, list) or len(turns) == 0:
            return ["発言が無い"]
        for index in range(len(turns)):
            turn = turns[index]
            if not isinstance(turn, dict):
                problems.append(f"{index} 番目が辞書でない")
                continue
            role = turn.get("role")
            content = turn.get("content")
            if role not in ("user", "assistant"):
                problems.append(f"{index} 番目の role が不正({role})")
            if not isinstance(content, str) or content.strip() == "":
                problems.append(f"{index} 番目が空")
                continue
            if "has_answer" in turn and not isinstance(turn["has_answer"], bool):
                problems.append(f"{index} 番目の has_answer が true か false でない")
            letters = len(LATIN.findall(content))
            # 短い発言(チェスの指し手、型番、コードの一語など)は英字でも正しいことが多いので見ない
            if len(content) > 10 and letters / len(content) > self.max_latin and not JAPANESE.search(content):
                problems.append(f"{index} 番目が英語のまま")
            if FENCE.search(content) and content.strip().startswith("```json"):
                problems.append(f"{index} 番目に JSON の囲みが残っている")
            if META_WORDS.search(content):
                problems.append(f"{index} 番目に作り直しの言葉の疑い")
        last = turns[-1]
        if isinstance(last, dict) and isinstance(last.get("content"), str) and last["content"].strip() != "":
            tail = last["content"].rstrip()
            # 絵文字や記号(Unicode の So)で終わるものは、文の終わりとして扱う
            symbol_end = unicodedata.category(tail[-1]) in ("So", "Sk", "Sm")
            if tail.endswith("、") or tail.endswith(",") or not (symbol_end or ENDINGS.search(tail)):
                problems.append("最後の発言が文の途中で終わっている(途中で切れた疑い)")
        return problems

    @staticmethod
    def ratio(turns, source_turns):
        # 元と比べた字数の比(縮み具合の分布を報告するため。検査で落とすのには使わない)
        made = 0
        for turn in turns:
            if isinstance(turn, dict) and isinstance(turn.get("content"), str):
                made += len(turn["content"])
        original = 0
        for turn in source_turns:
            original += len(turn["content"])
        if original == 0:
            return None
        return made / original


class FileChecker:
    def __init__(self):
        self.turns = TurnChecker()
        self.ratios = []

    def item(self, row, source=None):
        problems = []
        for key, kind in FIELDS.items():
            if key not in row:
                problems.append(f"{key} が無い")
            elif not isinstance(row[key], kind):
                problems.append(f"{key} の型が違う")
        if problems:
            return problems
        n = len(row["haystack_sessions"])
        if len(row["haystack_session_ids"]) != n or len(row["haystack_dates"]) != n:
            problems.append("会話の番号、日付、会話の本数がそろっていない")
        if len(set(row["haystack_session_ids"])) != len(row["haystack_session_ids"]):
            problems.append("会話の番号が重なっている")
        for sid in row["answer_session_ids"]:
            if sid not in row["haystack_session_ids"]:
                problems.append(f"根拠の会話 {sid} が履歴に無い")
        if str(row["question"]).strip() == "" or str(row["answer"]).strip() == "":
            problems.append("質問か答えが空")
        for index in range(n):
            source_turns = None
            if source is not None:
                source_turns = source["haystack_sessions"][index]
            for p in self.turns.turns(row["haystack_sessions"][index]):
                problems.append(f"{index} 本目の会話: {p}")
            if source_turns is not None:
                value = self.turns.ratio(row["haystack_sessions"][index], source_turns)
                if value is not None:
                    self.ratios.append(value)
        return problems

    def run(self, path, source_path=None):
        rows = json.loads(open(path, encoding="utf-8").read())
        sources = {}
        if source_path is not None:
            for row in json.loads(open(source_path, encoding="utf-8").read()):
                sources[row["question_id"]] = row
        report = {}
        ok = 0
        for row in rows:
            problems = self.item(row, sources.get(row.get("question_id")))
            report[row.get("question_id", "?")] = problems
            if not problems:
                ok += 1
        logger.info("形式の検査 %s: %d 問中 %d 問に問題なし", path, len(rows), ok)
        if self.ratios:
            ordered = sorted(self.ratios)
            logger.info("元と比べた字数の比(会話ごと): 中央値 %.2f、10%% 点 %.2f、最小 %.2f、最大 %.2f",
                        ordered[len(ordered) // 2], ordered[len(ordered) // 10], ordered[0], ordered[-1])
        return report
