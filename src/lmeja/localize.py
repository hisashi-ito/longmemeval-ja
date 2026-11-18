"""LongMemEval(英語)の問題を、日本の暮らしの日本語の会話に作り直す。

Claude の Message Batches API に 1 問 1 依頼で送り、結果を 1 問 1 ファイルで out/ に書き出す。
鍵は conf.yaml の key_file から読み込み、表示も記録もしない。
"""
import json
import logging
import os
import random
import time
from pathlib import Path

import anthropic
import yaml
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s -- : %(message)s')
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]


def load_conf():
    with open(ROOT / "conf.yaml", mode="r", encoding="utf-8") as f:
        params = yaml.load(f, Loader=yaml.SafeLoader)
    return params


def load_key():
    # 鍵は環境変数 ANTHROPIC_API_KEY で渡す。無ければ LMEJA_KEY_FILE の .env から ANTHROPIC_API_KEY= の行だけを読む。
    # 値は表示も記録もしない
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    path = os.environ.get("LMEJA_KEY_FILE")
    if not path:
        raise RuntimeError("ANTHROPIC_API_KEY か LMEJA_KEY_FILE を環境変数で渡す")
    with open(os.path.expanduser(path), mode="r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("ANTHROPIC_API_KEY="):
                os.environ["ANTHROPIC_API_KEY"] = line.split("=", 1)[1]
                return
    raise RuntimeError("鍵のファイルに ANTHROPIC_API_KEY の行が無い")


class Dataset:
    def __init__(self, params):
        self.params = params
        self.items = json.loads((ROOT / params["common"]["source"]).read_text(encoding="utf-8"))

    def select_trial(self):
        # 問題の種類ごとに決めた数を、決まった乱数で選ぶ。答えない判断の問題を決めた数だけ混ぜる
        conf = self.params["trial"]
        rng = random.Random(conf["seed"])
        abstention = []
        by_type = {}
        for item in self.items:
            if item["question_id"].endswith("_abs"):
                abstention.append(item)
            else:
                by_type.setdefault(item["question_type"], []).append(item)
        chosen = []
        for item in rng.sample(abstention, conf["abstention"]):
            chosen.append(item)
        for kind, count in conf["per_type"].items():
            need = count
            for item in chosen:
                if item["question_type"] == kind:
                    need -= 1
            for item in rng.sample(by_type[kind], max(0, need)):
                chosen.append(item)
        return chosen


class Localizer:
    def __init__(self, params):
        self.params = params
        conf = params["localize"]
        self.model = conf["model"]
        self.effort = conf["effort"]
        self.max_tokens = conf["max_tokens"]
        self.poll_seconds = conf["poll_seconds"]
        self.out_dir = ROOT / params["common"]["out_dir"]
        self.instruction = (Path(__file__).parent / "prompt.md").read_text(encoding="utf-8")
        load_key()
        self.client = anthropic.Anthropic()

    def payload(self, item):
        sessions = []
        for sid, date, turns in zip(item["haystack_session_ids"], item["haystack_dates"], item["haystack_sessions"]):
            sessions.append({"session_id": sid, "date": date, "turns": turns})
        return {
            "question_id": item["question_id"],
            "question_type": item["question_type"],
            "question": item["question"],
            "answer": item["answer"],
            "question_date": item["question_date"],
            "sessions": sessions,
        }

    def request(self, item):
        body = json.dumps(self.payload(item), ensure_ascii=False)
        params = MessageCreateParamsNonStreaming(
            model=self.model,
            max_tokens=self.max_tokens,
            output_config={"effort": self.effort},
            # 指示文は全問で同じなので、キャッシュに載せる
            system=[{"type": "text", "text": self.instruction, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": body}],
        )
        return Request(custom_id=item["question_id"], params=params)

    def submit(self, items, name):
        requests = []
        for item in items:
            requests.append(self.request(item))
        batch = self.client.messages.batches.create(requests=requests)
        record = {"batch_id": batch.id, "name": name, "count": len(requests), "created": time.time()}
        (self.out_dir / f"batch_{name}.json").write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
        logger.info("バッチを出した: %s(%d 問)", batch.id, len(requests))
        return batch.id

    def wait(self, batch_id):
        while True:
            batch = self.client.messages.batches.retrieve(batch_id)
            counts = batch.request_counts
            logger.info("状態 %s 処理中 %d 成功 %d 失敗 %d", batch.processing_status, counts.processing,
                        counts.succeeded, counts.errored)
            if batch.processing_status == "ended":
                return batch
            time.sleep(self.poll_seconds)

    def fetch(self, batch_id, name):
        result_dir = self.out_dir / name
        result_dir.mkdir(parents=True, exist_ok=True)
        usage_total = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0,
                       "cache_creation_input_tokens": 0}
        for result in self.client.messages.batches.results(batch_id):
            qid = result.custom_id
            if result.result.type != "succeeded":
                logger.warning("%s: %s", qid, result.result.type)
                (result_dir / f"{qid}.error.json").write_text(result.to_json(), encoding="utf-8")
                continue
            message = result.result.message
            text = ""
            for block in message.content:
                if block.type == "text":
                    text += block.text
            usage = message.usage
            for key in usage_total:
                value = getattr(usage, key, 0) or 0
                usage_total[key] += value
            (result_dir / f"{qid}.raw.txt").write_text(text, encoding="utf-8")
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                logger.warning("%s: JSON として読めない(stop_reason=%s)", qid, message.stop_reason)
                continue
            if not isinstance(parsed, dict):
                # まれに JSON の配列で返る。読めなかったものとして扱い、出し直しに回す
                logger.warning("%s: JSON の形が違う(%s)", qid, type(parsed).__name__)
                continue
            parsed["question_id"] = qid
            (result_dir / f"{qid}.json").write_text(json.dumps(parsed, ensure_ascii=False, indent=1), encoding="utf-8")
        # 分けて送ったバッチごとに使ったトークンを残す
        (result_dir / f"_usage_{batch_id}.json").write_text(json.dumps(usage_total, indent=1), encoding="utf-8")
        logger.info("使ったトークン: %s", usage_total)
        return usage_total


class ChunkedRun:
    """多数の依頼を、決まった本数ずつのバッチに分けて送り、全部の終わりを待って結果を落とす。"""

    def __init__(self, localizer, chunk):
        self.localizer = localizer
        self.chunk = chunk

    def run(self, requests, name):
        # 1 本ずつ出し、終わって結果を落としてから次を出す(2026-11-18、4,000 件ずつまとめて出したら 1 時間たっても
        # 中身が見えず、依頼者の判断で小さく順に出す形にした。バッチの大きさは費用に関係しない)
        out_dir = self.localizer.out_dir
        record_path = out_dir / f"batch_{name}.json"
        batch_ids = []
        if record_path.exists():
            batch_ids = json.loads(record_path.read_text(encoding="utf-8")).get("batch_ids", [])
        start = 0
        while start < len(requests):
            part = requests[start:start + self.chunk]
            batch = self.localizer.client.messages.batches.create(requests=part)
            batch_ids.append(batch.id)
            record_path.write_text(json.dumps({"batch_ids": batch_ids}, indent=1), encoding="utf-8")
            logger.info("バッチを出した: %s(%d 件、%d / %d 件目から)", batch.id, len(part), start, len(requests))
            self.localizer.wait(batch.id)
            self.localizer.fetch(batch.id, name)
            start += self.chunk
        return batch_ids
