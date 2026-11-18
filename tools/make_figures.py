"""README に載せるデータの図を作る。英語の図は figures/en/、日本語の図は figures/ja/ に出す。

    uv run python tools/make_figures.py

入力: out/longmemeval_ja_oracle.json、out/longmemeval_ja_s.json(あれば)、data/raw/ の英語の元、
out/verify_full_{ja,en}.json(あれば)。無いものの図は作らない。
"""
import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"

# 色は役割で固定する(英語の元 = 青、日本語の作り直し = 橙、3 つめ = 水色)。色覚の検査は通してある
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
INK = "#0b0b0b"
MUTED = "#52514e"

TYPES = ["temporal-reasoning", "multi-session", "knowledge-update", "single-session-user",
         "single-session-assistant", "single-session-preference"]

TEXT = {
    "en": {
        "types": {"temporal-reasoning": "Temporal reasoning", "multi-session": "Multi-session",
                  "knowledge-update": "Knowledge update", "single-session-user": "Single-session (user)",
                  "single-session-assistant": "Single-session (assistant)",
                  "single-session-preference": "Single-session (preference)"},
        "questions": "Questions",
        "abstention": "of which abstention",
        "history_chars": "Characters per history (thousands)",
        "histories": "Questions",
        "english": "English original",
        "japanese": "Japanese remake",
        "sessions_per_q": "Sessions per question",
        "sources": {"longmemeval": "LongMemEval authors", "ultrachat": "UltraChat", "new": "Newly written (ShareGPT slots)",
                    "evidence": "Evidence sessions"},
        "sessions": "Unique sessions",
        "position": "Position of evidence in the history (0 = oldest, 1 = newest)",
        "evidence_n": "Evidence sessions",
        "accuracy": "Accuracy (%)",
        "ratio": "Japanese characters / English characters (per session)",
        "sessions_n": "Sessions",
    },
    "ja": {
        "types": {"temporal-reasoning": "時間の推論", "multi-session": "複数の会話をまたぐ", "knowledge-update": "知識の更新",
                  "single-session-user": "利用者が話した事実", "single-session-assistant": "アシスタントが話した事実",
                  "single-session-preference": "好み"},
        "questions": "問題数",
        "abstention": "うち答えない判断",
        "history_chars": "1 問あたりの履歴の字数(千字)",
        "histories": "問題数",
        "english": "英語の元",
        "japanese": "日本語の作り直し",
        "sessions_per_q": "1 問あたりの会話の本数",
        "sources": {"longmemeval": "LongMemEval の著者の会話", "ultrachat": "UltraChat",
                    "new": "新しく書いた会話(ShareGPT の位置)", "evidence": "根拠の会話"},
        "sessions": "会話の本数(重なりを除く)",
        "position": "履歴の中の根拠の位置(0 = いちばん古い、1 = いちばん新しい)",
        "evidence_n": "根拠の会話の本数",
        "accuracy": "正解の割合(%)",
        "ratio": "日本語の字数 / 英語の字数(会話ごと)",
        "sessions_n": "会話の本数",
    },
}


class Figures:
    def __init__(self):
        plt.rcParams.update({
            "font.family": ["DejaVu Sans", "Noto Sans CJK JP"],
            "axes.labelsize": 13, "axes.titlesize": 13, "xtick.labelsize": 11, "ytick.labelsize": 11,
            "legend.fontsize": 11, "axes.spines.top": False, "axes.spines.right": False,
            "legend.frameon": False, "axes.grid": True, "grid.alpha": 0.2, "grid.linewidth": 0.6,
            "axes.axisbelow": True, "savefig.dpi": 200, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
            "xtick.color": MUTED, "ytick.color": MUTED,
        })
        self.oracle = self.load(OUT / "longmemeval_ja_oracle.json")
        self.s_ja = self.load(OUT / "longmemeval_ja_s.json")
        self.s_en = self.load(ROOT / "data" / "raw" / "longmemeval_s_cleaned.json")
        self.oracle_en = self.load(ROOT / "data" / "raw" / "longmemeval_oracle.json")
        self.verify_ja = self.load(OUT / "verify_final_ja.json")
        self.verify_en = self.load(OUT / "verify_full_en.json")

    @staticmethod
    def load(path):
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def save(fig, lang, name):
        path = ROOT / "figures" / lang / (name + ".png")
        fig.tight_layout()
        fig.savefig(path, facecolor="white")
        plt.close(fig)
        print("wrote", path.relative_to(ROOT))

    @staticmethod
    def chars(sessions):
        total = 0
        for session in sessions:
            for turn in session:
                total += len(turn["content"])
        return total

    def question_types(self, lang):
        t = TEXT[lang]
        counts = []
        abstentions = []
        for kind in TYPES:
            n = 0
            a = 0
            for row in self.oracle:
                if row["question_type"] == kind:
                    n += 1
                    if row["question_id"].endswith("_abs"):
                        a += 1
            counts.append(n)
            abstentions.append(a)
        fig, ax = plt.subplots(figsize=(8, 4))
        labels = []
        for kind in TYPES:
            labels.append(t["types"][kind])
        y = list(range(len(TYPES)))
        ax.barh(y, counts, color=BLUE, height=0.6)
        for i in range(len(TYPES)):
            note = str(counts[i])
            if abstentions[i] > 0:
                note += f"  ({t['abstention']} {abstentions[i]})"
            ax.text(counts[i] + 2, i, note, va="center", color=INK, fontsize=11)
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_xlabel(t["questions"])
        ax.set_xlim(0, max(counts) * 1.75)
        ax.grid(axis="y", visible=False)
        self.save(fig, lang, "question_types")

    def history_length(self, lang):
        if self.s_ja is None:
            return
        t = TEXT[lang]
        ids = set()
        for row in self.s_ja:
            ids.add(row["question_id"])
        ja = []
        for row in self.s_ja:
            ja.append(self.chars(row["haystack_sessions"]) / 1000)
        en = []
        for row in self.s_en:
            if row["question_id"] in ids:
                en.append(self.chars(row["haystack_sessions"]) / 1000)
        fig, ax = plt.subplots(figsize=(8, 4))
        top = max(max(en), max(ja))
        bins = []
        step = top / 40
        for i in range(41):
            bins.append(i * step)
        ax.hist(en, bins=bins, color=BLUE, alpha=0.85, label=f"{t['english']} (median {statistics.median(en):.0f}k)"
                if lang == "en" else f"{t['english']}(中央値 {statistics.median(en):.0f} 千字)")
        ax.hist(ja, bins=bins, color=ORANGE, alpha=0.85, label=f"{t['japanese']} (median {statistics.median(ja):.0f}k)"
                if lang == "en" else f"{t['japanese']}(中央値 {statistics.median(ja):.0f} 千字)")
        ax.set_xlabel(t["history_chars"])
        ax.set_ylabel(t["histories"])
        ax.legend()
        self.save(fig, lang, "history_length")

    def length_ratio(self, lang):
        t = TEXT[lang]
        en = {}
        for row in self.oracle_en:
            for sid, turns in zip(row["haystack_session_ids"], row["haystack_sessions"]):
                en[sid] = self.chars([turns])
        ratios = []
        seen = set()
        for row in self.oracle:
            for sid, turns in zip(row["haystack_session_ids"], row["haystack_sessions"]):
                if sid in seen or en.get(sid, 0) == 0:
                    continue
                seen.add(sid)
                ratios.append(self.chars([turns]) / en[sid])
        fig, ax = plt.subplots(figsize=(8, 3.6))
        bins = []
        for i in range(36):
            bins.append(i * 0.025)
        ax.hist(ratios, bins=bins, color=ORANGE)
        median = statistics.median(ratios)
        ax.axvline(median, color=INK, linewidth=1, linestyle="--")
        ax.text(median + 0.01, ax.get_ylim()[1] * 0.9, ("median " if lang == "en" else "中央値 ") + f"{median:.2f}",
                color=INK, fontsize=11)
        ax.set_xlabel(t["ratio"])
        ax.set_ylabel(t["sessions_n"])
        self.save(fig, lang, "length_ratio")

    def sources(self, lang):
        if self.s_ja is None:
            return
        t = TEXT[lang]
        evidence = set()
        for row in self.s_en:
            for sid in row["answer_session_ids"]:
                evidence.add(sid)
        counts = {"evidence": set(), "longmemeval": set(), "ultrachat": set(), "new": set()}
        for row in self.s_ja:
            for sid in row["haystack_session_ids"]:
                if sid in evidence:
                    counts["evidence"].add(sid)
                elif sid.startswith("sharegpt"):
                    counts["new"].add(sid)
                elif sid.startswith("ultrachat"):
                    counts["ultrachat"].add(sid)
                else:
                    counts["longmemeval"].add(sid)
        keys = ["evidence", "longmemeval", "ultrachat", "new"]
        colors = [MUTED, BLUE, ORANGE, AQUA]
        values = []
        labels = []
        for key in keys:
            values.append(len(counts[key]))
            labels.append(t["sources"][key])
        fig, ax = plt.subplots(figsize=(8, 3.4))
        y = list(range(len(keys)))
        ax.barh(y, values, color=colors, height=0.6)
        for i in range(len(keys)):
            ax.text(values[i] + 80, i, f"{values[i]:,}", va="center", color=INK, fontsize=11)
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_xlabel(t["sessions"])
        ax.set_xlim(0, max(values) * 1.2)
        ax.grid(axis="y", visible=False)
        self.save(fig, lang, "sources")

    def evidence_position(self, lang):
        if self.s_ja is None:
            return
        t = TEXT[lang]
        positions = []
        for row in self.s_ja:
            n = len(row["haystack_session_ids"])
            if n < 2:
                continue
            for sid in row["answer_session_ids"]:
                index = row["haystack_session_ids"].index(sid)
                positions.append(index / (n - 1))
        fig, ax = plt.subplots(figsize=(8, 3.4))
        bins = []
        for i in range(21):
            bins.append(i / 20)
        ax.hist(positions, bins=bins, color=BLUE)
        ax.set_xlabel(t["position"])
        ax.set_ylabel(t["evidence_n"])
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        self.save(fig, lang, "evidence_position")

    def verification(self, lang):
        if self.verify_ja is None or self.verify_en is None:
            return
        t = TEXT[lang]
        ja = self.verify_ja["report"]
        en = self.verify_en["report"]
        rate_ja = []
        rate_en = []
        for kind in TYPES:
            total = 0
            ok_ja = 0
            ok_en = 0
            for qid, row in ja.items():
                if row["type"] != kind or qid not in en:
                    continue
                total += 1
                ok_ja += 1 if row["passed"] else 0
                ok_en += 1 if en[qid]["passed"] else 0
            rate_ja.append(100 * ok_ja / total if total else 0)
            rate_en.append(100 * ok_en / total if total else 0)
        fig, ax = plt.subplots(figsize=(9, 4.2))
        x = list(range(len(TYPES)))
        width = 0.38
        left = []
        right = []
        for i in x:
            left.append(i - width / 2 - 0.01)
            right.append(i + width / 2 + 0.01)
        ax.bar(left, rate_en, width, color=BLUE, label=t["english"])
        ax.bar(right, rate_ja, width, color=ORANGE, label=t["japanese"])
        for i in x:
            ax.text(left[i], rate_en[i] + 1, f"{rate_en[i]:.0f}", ha="center", fontsize=9, color=INK)
            ax.text(right[i], rate_ja[i] + 1, f"{rate_ja[i]:.0f}", ha="center", fontsize=9, color=INK)
        labels = []
        for kind in TYPES:
            if lang == "en":
                labels.append(t["types"][kind].replace(" (", "\n("))
            else:
                # 日本語のラベルは長いので、「が」「を」のあとで 2 行に割る(隣どうしで重ならないように)
                label = t["types"][kind]
                for mark in ("が", "を"):
                    if mark in label:
                        label = label.replace(mark, mark + "\n", 1)
                        break
                labels.append(label)
        ax.set_xticks(x, labels, fontsize=9)
        ax.set_ylabel(t["accuracy"])
        ax.set_ylim(0, 110)
        # 凡例は棒にかぶらないよう、図の上に横並びで置く
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
        ax.grid(axis="x", visible=False)
        self.save(fig, lang, "verification")

    def run(self):
        if self.oracle is None:
            sys.exit("out/longmemeval_ja_oracle.json が無い")
        for lang in ("en", "ja"):
            self.question_types(lang)
            self.length_ratio(lang)
            self.history_length(lang)
            self.sources(lang)
            self.evidence_position(lang)
            self.verification(lang)


if __name__ == "__main__":
    Figures().run()
