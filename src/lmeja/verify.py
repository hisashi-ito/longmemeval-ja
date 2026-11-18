"""作り直した問題で、答えが会話から導けるかを確かめる。

1. 解く役: 日本語の会話(日付つき)と質問だけを渡して解かせる。正解は見せない
2. 判定する役: 解いた答えと正解を並べて、合っているかを判定させる。元の LongMemEval の採点(evaluate_qa.py)と
   同じく、問題の種類ごとに判定の基準を変える
解く役が正解にたどり着けなかった問題は、作り直しで答えの根拠が崩れた疑いがあるので、人が見る。
"""
import json
import logging
from pathlib import Path

from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from lmeja.localize import ROOT, Localizer

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s -- : %(message)s')
logger = logging.getLogger(__name__)

SOLVE_INSTRUCTION = """あなたは、利用者と AI アシスタントの過去の会話の記録を読んで、利用者の質問に答えるアシスタントです。
会話の記録は日付つきで、古い順に並んでいます。質問した日付は、質問の頭に書いてあります。
記録から分かることだけで答えてください。記録から答えが分からないときは、分からないと答えてください。
答えは、質問に直接答える短い日本語で書きます。"""

JUDGE_BASE = """次の質問について、正解と、ある回答が与えられます。回答が正解と同じ内容か判定してください。
{rule}
「はい」か「いいえ」だけで答えてください。

質問: {question}
正解: {answer}
回答: {hypothesis}"""

JUDGE_RULES = {
    "default": "回答に正解が含まれていれば「はい」。正解と同じ意味の言い換えも「はい」。途中の考えだけで正解に至っていなければ「いいえ」。",
    "temporal-reasoning": "回答に正解が含まれていれば「はい」。日数、週数、月数の数え方で 1 だけずれる(数え始めの日を含めるかどうか)ものは「はい」とする。",
    "knowledge-update": "回答に正解(いまの情報)が含まれていれば「はい」。古い情報も一緒に書いてあっても、いまの情報として正解を答えていれば「はい」。",
    "single-session-preference": "正解は、利用者が好みそうな答えの目安です。回答が、その目安に書かれた利用者の事情や好みをきちんと踏まえていれば「はい」。全部を満たさなくてよい。",
    "abstention": "正解は「その情報は会話に無い」という趣旨です。回答が、分からない、話されていない、と答えていれば「はい」。何か具体的に答えてしまっていれば「いいえ」。",
}


class Verifier(Localizer):
    def __init__(self, params):
        super().__init__(params)
        conf = params["verify"]
        self.solve_model = conf["solve_model"]
        self.solve_effort = conf["solve_effort"]
        self.judge_model = conf["judge_model"]

    @staticmethod
    def history_text(made):
        lines = []
        for session in made["sessions"]:
            lines.append(f"### {session['date']} の会話")
            for turn in session["turns"]:
                speaker = "利用者" if turn["role"] == "user" else "アシスタント"
                lines.append(f"{speaker}: {turn['content']}")
            lines.append("")
        return "\n".join(lines)

    def solve_request(self, made):
        body = self.history_text(made) + f"\n### 質問({made['question_date']})\n{made['question']}"
        params = MessageCreateParamsNonStreaming(
            model=self.solve_model,
            max_tokens=16000,
            output_config={"effort": self.solve_effort},
            system=SOLVE_INSTRUCTION,
            messages=[{"role": "user", "content": body}],
        )
        return Request(custom_id=made["question_id"], params=params)

    def judge_request(self, made, question_type, hypothesis):
        kind = question_type
        if made["question_id"].endswith("_abs"):
            kind = "abstention"
        rule = JUDGE_RULES.get(kind, JUDGE_RULES["default"])
        text = JUDGE_BASE.format(rule=rule, question=made["question"], answer=made["answer"], hypothesis=hypothesis)
        params = MessageCreateParamsNonStreaming(
            model=self.judge_model,
            max_tokens=2000,
            output_config={"effort": "low"},
            messages=[{"role": "user", "content": text}],
        )
        return Request(custom_id=made["question_id"], params=params)

    @staticmethod
    def passed(verdict):
        # 判定の答えは「はい」か「いいえ」。太字や句点、前置きが付くことがあるので、印を外してから見る
        text = verdict.replace("*", "").replace("「", "").replace("」", "").strip()
        if "いいえ" in text[:10]:
            return False
        return text.startswith("はい")

    def submit_requests(self, requests, name):
        batch = self.client.messages.batches.create(requests=requests)
        record = {"batch_id": batch.id, "name": name, "count": len(requests)}
        (self.out_dir / f"batch_{name}.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
        logger.info("バッチを出した: %s(%d 件)", batch.id, len(requests))
        return batch.id

    def texts(self, batch_id):
        # バッチの結果を、番号ごとの本文の文字列にする(JSON として読まない)
        found = {}
        usage = {"input_tokens": 0, "output_tokens": 0}
        for result in self.client.messages.batches.results(batch_id):
            if result.result.type != "succeeded":
                logger.warning("%s: %s", result.custom_id, result.result.type)
                continue
            message = result.result.message
            text = ""
            for block in message.content:
                if block.type == "text":
                    text += block.text
            found[result.custom_id] = text.strip()
            usage["input_tokens"] += message.usage.input_tokens
            usage["output_tokens"] += message.usage.output_tokens
        logger.info("使ったトークン: %s", usage)
        return found, usage

    def run(self, made_items, types, name):
        solve_requests = []
        for made in made_items:
            solve_requests.append(self.solve_request(made))
        solve_batch = self.submit_requests(solve_requests, name + "_solve")
        self.wait(solve_batch)
        hypotheses, solve_usage = self.texts(solve_batch)
        requests = []
        for made in made_items:
            if made["question_id"] in hypotheses:
                requests.append(self.judge_request(made, types[made["question_id"]], hypotheses[made["question_id"]]))
        judge_batch = self.submit_requests(requests, name + "_judge")
        self.wait(judge_batch)
        verdicts, judge_usage = self.texts(judge_batch)
        report = {}
        ok = 0
        for made in made_items:
            qid = made["question_id"]
            verdict = verdicts.get(qid, "")
            passed = self.passed(verdict)
            if passed:
                ok += 1
            report[qid] = {"type": types[qid], "question": made["question"], "answer": made["answer"],
                           "hypothesis": hypotheses.get(qid, ""), "verdict": verdict, "passed": passed}
        logger.info("答えの確かめ: %d 問中 %d 問で、会話から正解が導けた", len(made_items), ok)
        out = self.out_dir / f"{name}.json"
        out.write_text(json.dumps({"solve_usage": solve_usage, "judge_usage": judge_usage, "report": report},
                                  ensure_ascii=False, indent=1), encoding="utf-8")
        return report
