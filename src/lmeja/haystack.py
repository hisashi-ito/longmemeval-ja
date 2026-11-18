"""S 版の履歴に挟まる会話(答えに関係しない会話)を、日本語に作り直す。

対象は、LongMemEval の著者が作った会話と UltraChat の会話(どちらも MIT)。ShareGPT の会話は、実在の利用者の
会話なので、訳さない。1 本 1 依頼で Message Batches API に送り、結果を 1 本 1 ファイルで書き出す。
"""
import json
import logging
import random
from pathlib import Path

from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from lmeja.localize import ROOT, Localizer

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s -- : %(message)s')
logger = logging.getLogger(__name__)


class FillerPool:
    def __init__(self, params):
        self.params = params
        source = ROOT / params["haystack"]["source"]
        items = json.loads(source.read_text(encoding="utf-8"))
        evidence = set()
        for item in items:
            for sid in item["answer_session_ids"]:
                evidence.add(sid)
        # 重なりを除き、根拠でも ShareGPT でもない会話を集める
        self.sessions = {}
        for item in items:
            for sid, turns in zip(item["haystack_session_ids"], item["haystack_sessions"]):
                if sid in evidence or sid.startswith("sharegpt") or sid in self.sessions:
                    continue
                self.sessions[sid] = turns

    @staticmethod
    def origin(sid):
        if sid.startswith("ultrachat"):
            return "ultrachat"
        return "longmemeval"

    def sample(self, count, seed):
        # 出どころの割合を保って選ぶ
        rng = random.Random(seed)
        ids = sorted(self.sessions)
        return rng.sample(ids, count)


class FillerLocalizer(Localizer):
    def __init__(self, params):
        super().__init__(params)
        conf = params["haystack"]
        self.model = conf["model"]
        self.effort = conf["effort"]
        self.max_tokens = conf["max_tokens"]
        self.instruction = (Path(__file__).parent / "filler_prompt.md").read_text(encoding="utf-8")

    def filler_request(self, sid, turns):
        body = json.dumps({"session_id": sid, "turns": turns}, ensure_ascii=False)
        params = MessageCreateParamsNonStreaming(
            model=self.model,
            max_tokens=self.max_tokens,
            output_config={"effort": self.effort},
            system=[{"type": "text", "text": self.instruction, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": body}],
        )
        return Request(custom_id=sid, params=params)

    def submit_fillers(self, pool, ids, name):
        requests = []
        for sid in ids:
            requests.append(self.filler_request(sid, pool.sessions[sid]))
        batch = self.client.messages.batches.create(requests=requests)
        record = {"batch_id": batch.id, "name": name, "count": len(requests)}
        (self.out_dir / f"batch_{name}.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
        logger.info("バッチを出した: %s(%d 本)", batch.id, len(requests))
        return batch.id


class FillerChecker:
    def __init__(self, pool):
        self.pool = pool

    def run(self, result_dir):
        ok = 0
        total = 0
        report = {}
        for path in sorted(Path(result_dir).glob("*.json")):
            if path.name.startswith("_"):
                continue
            made = json.loads(path.read_text(encoding="utf-8"))
            sid = made["question_id"]
            source = self.pool.sessions[sid]
            turns = made.get("turns", [])
            problems = []
            if len(turns) != len(source):
                problems.append(f"発言の数が違う({len(source)} → {len(turns)})")
            else:
                for t in range(len(turns)):
                    if turns[t].get("role") != source[t]["role"]:
                        problems.append(f"{t} 番目の role が違う")
            total += 1
            if not problems:
                ok += 1
            report[sid] = problems
        logger.info("挟まる会話の形の確かめ: %d 本中 %d 本が元と同じ形", total, ok)
        (Path(result_dir) / "_shape.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        return report
