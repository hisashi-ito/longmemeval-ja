"""作り直した問題の形を、元の問題と突き合わせて確かめる(LLM を使わない)。

会話の本数と順番、各会話の発言の数と順番、role、has_answer の位置、日付が元と同じか。
答えが会話から導けるかの確かめ(LLM による)は、別の段で行う。
"""
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s -- : %(message)s')
logger = logging.getLogger(__name__)


class ShapeChecker:
    def __init__(self, source_items):
        self.source = {}
        for item in source_items:
            self.source[item["question_id"]] = item

    def check(self, made):
        problems = []
        original = self.source[made["question_id"]]
        if made.get("question_date") != original["question_date"]:
            problems.append("質問の日付が変わった")
        sessions = made.get("sessions", [])
        if len(sessions) != len(original["haystack_sessions"]):
            problems.append(f"会話の本数が違う({len(original['haystack_sessions'])} → {len(sessions)})")
            return problems
        for index in range(len(sessions)):
            mine = sessions[index]
            sid = original["haystack_session_ids"][index]
            if mine.get("session_id") != sid:
                problems.append(f"{index} 本目の会話の番号が違う")
            if mine.get("date") != original["haystack_dates"][index]:
                problems.append(f"{index} 本目の会話の日付が違う")
            turns = mine.get("turns", [])
            source_turns = original["haystack_sessions"][index]
            if len(turns) != len(source_turns):
                problems.append(f"{index} 本目の発言の数が違う({len(source_turns)} → {len(turns)})")
                continue
            for t in range(len(turns)):
                if turns[t].get("role") != source_turns[t]["role"]:
                    problems.append(f"{index} 本目 {t} 番目の role が違う")
                if bool(turns[t].get("has_answer")) != bool(source_turns[t].get("has_answer")):
                    problems.append(f"{index} 本目 {t} 番目の has_answer が違う")
        return problems

    def restore_marks(self, made):
        # 根拠の印(has_answer)は中身ではなく目印なので、会話と発言の数が元と同じなら、元の位置に戻す
        original = self.source[made["question_id"]]
        sessions = made.get("sessions", [])
        if len(sessions) != len(original["haystack_sessions"]):
            return False
        for index in range(len(sessions)):
            if len(sessions[index].get("turns", [])) != len(original["haystack_sessions"][index]):
                return False
        changed = False
        for index in range(len(sessions)):
            turns = sessions[index]["turns"]
            source_turns = original["haystack_sessions"][index]
            for t in range(len(turns)):
                want = bool(source_turns[t].get("has_answer"))
                if bool(turns[t].get("has_answer")) != want:
                    changed = True
                    if want:
                        turns[t]["has_answer"] = True
                    else:
                        turns[t].pop("has_answer", None)
        return changed

    def run(self, result_dir):
        report = {}
        for path in sorted(Path(result_dir).glob("*.json")):
            if path.name.startswith("_") or path.name.endswith(".error.json"):
                continue
            made = json.loads(path.read_text(encoding="utf-8"))
            if self.restore_marks(made):
                logger.info("%s: 根拠の印を元の位置に戻した", made["question_id"])
                path.write_text(json.dumps(made, ensure_ascii=False, indent=1), encoding="utf-8")
            report[made["question_id"]] = self.check(made)
        ok = 0
        for qid, problems in report.items():
            if not problems:
                ok += 1
            else:
                logger.warning("%s: %s", qid, " / ".join(problems))
        logger.info("形の確かめ: %d 問中 %d 問が元と同じ形", len(report), ok)
        (Path(result_dir) / "_shape.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        return report
