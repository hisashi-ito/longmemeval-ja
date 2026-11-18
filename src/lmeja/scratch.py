"""S 版で ShareGPT の会話が入っていた位置に入れる会話を、日本語でゼロから書かせる。

ShareGPT の会話の中身は使わない。借りるのは、会話ごとの発言の数、長さ、最初の話し手の数字だけ。
話題の一覧を LLM に挙げさせ、話題ごとに 1 本の会話を Message Batches API で書かせる。
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

TOPIC_INSTRUCTION = """日本で暮らし、働く人が、AI アシスタントに相談したり頼んだりしそうな話題を挙げてください。
分野を {categories} 個、各分野に {per_category} 個ずつ挙げます。分野は、料理、家事、家電、住まい、旅行、交通、
お金と税、制度と手続き、健康と運動、勉強と資格、語学、プログラミング、IT とパソコン、文章の手直し、仕事の進め方、
趣味、文化と歴史、科学の素朴な疑問、など幅広くします。話題は具体的な 1 文にします(例: 「確定申告で医療費控除を
受けるときに必要な書類」)。利用者自身の持ち物や出来事の話ではなく、一般的な質問や作業の依頼にします。

次の形の JSON だけを出力します。前置きや説明、コードの囲みは付けません。
{{"topics": [{{"category": "分野", "topic": "話題"}}]}}"""


class SlotShapes:
    def __init__(self, params):
        source = ROOT / params["haystack"]["source"]
        items = json.loads(source.read_text(encoding="utf-8"))
        # ShareGPT の会話の、発言の数、文字数、最初の話し手だけを集める(中身は保持しない)
        self.shapes = {}
        for item in items:
            for sid, turns in zip(item["haystack_session_ids"], item["haystack_sessions"]):
                if not sid.startswith("sharegpt") or sid in self.shapes:
                    continue
                chars = 0
                for turn in turns:
                    chars += len(turn["content"])
                first = turns[0]["role"] if turns else "user"
                self.shapes[sid] = {"turns": len(turns), "chars": chars, "first_role": first}


HINT_INSTRUCTION = """次は、利用者と AI の英語の会話です。この会話で利用者が AI に何を頼んだかを、一般化した日本語の 1 行の
話題にしてください。新しい日本語の会話を書くためのヒントにします。

- 名前、会社名、製品名、住所、数字、日付、固有の状況は書かない。誰の会話か分からない一般的な話題にする
- 会話の文を写さない。訳さない
- 日本の暮らしや仕事で自然な話題に置き換えてよい(例: 米国の税制の質問 → 日本の確定申告の質問)
- 頼み方の種類を次から 1 つ選ぶ: 質問、説明の依頼、文章を書く依頼、文章の手直し、コードを書く依頼、コードの直し、
  アイデア出し、計画づくり、翻訳、要約、ロールプレイ、その他

次の形の JSON だけを出力する。前置きや説明、コードの囲みは付けない。
{"category": "分野", "topic": "一般化した話題", "request_type": "頼み方の種類"}"""


class HintMaker(Localizer):
    """ShareGPT の会話から、一般化した 1 行の話題だけを取り出す。会話の文は出力に残さない。"""

    def __init__(self, params):
        super().__init__(params)
        conf = params["scratch"]
        self.model = conf["model"]
        self.effort = conf["effort"]
        self.max_tokens = 2000
        self.head_chars = conf["hint_head_chars"]

    def hint_request(self, slot_id, turns):
        # 話題が分かれば足りるので、頭の部分だけを渡す
        text = ""
        for turn in turns:
            text += "[" + turn["role"] + "] " + turn["content"] + "\n"
            if len(text) > self.head_chars:
                break
        params = MessageCreateParamsNonStreaming(
            model=self.model,
            max_tokens=self.max_tokens,
            output_config={"effort": self.effort},
            system=[{"type": "text", "text": HINT_INSTRUCTION, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": text[:self.head_chars]}],
        )
        return Request(custom_id=slot_id, params=params)

    def submit_hints(self, source_sessions, slot_ids, name):
        requests = []
        for slot_id in slot_ids:
            requests.append(self.hint_request(slot_id, source_sessions[slot_id]))
        batch = self.client.messages.batches.create(requests=requests)
        record = {"batch_id": batch.id, "name": name, "count": len(requests)}
        (self.out_dir / ("batch_" + name + ".json")).write_text(json.dumps(record, indent=1), encoding="utf-8")
        logger.info("バッチを出した: %s(%d 本)", batch.id, len(requests))
        return batch.id


class ScratchMaker(Localizer):
    def __init__(self, params):
        super().__init__(params)
        conf = params["scratch"]
        self.model = conf["model"]
        self.effort = conf["effort"]
        self.max_tokens = conf["max_tokens"]
        # 英語の文字数を、日本語のおおよその文字数に直す割合
        self.ja_per_en = conf["ja_chars_per_en_char"]
        self.max_turns = conf["max_turns"]
        self.instruction = (Path(__file__).parent / "scratch_prompt.md").read_text(encoding="utf-8")

    def make_topics(self, categories, per_category, path):
        text = TOPIC_INSTRUCTION.format(categories=categories, per_category=per_category)
        # 出力が長いので、ストリーミングで受けて最後の答えだけを使う
        with self.client.messages.stream(
            model=self.params["scratch"]["topic_model"],
            max_tokens=32000,
            messages=[{"role": "user", "content": text}],
        ) as stream:
            response = stream.get_final_message()
        body = ""
        for block in response.content:
            if block.type == "text":
                body += block.text
        topics = json.loads(body)["topics"]
        path.write_text(json.dumps(topics, ensure_ascii=False, indent=1), encoding="utf-8")
        logger.info("話題を %d 個作った", len(topics))
        return topics

    def scratch_request(self, slot_id, shape, topic):
        turns = min(shape["turns"], self.max_turns)
        spec = {
            "category": topic["category"],
            "topic": topic["topic"],
            "turns": turns,
            "chars": int(shape["chars"] * self.ja_per_en),
            "first_role": shape["first_role"],
        }
        params = MessageCreateParamsNonStreaming(
            model=self.model,
            max_tokens=self.max_tokens,
            output_config={"effort": self.effort},
            system=[{"type": "text", "text": self.instruction, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": json.dumps(spec, ensure_ascii=False)}],
        )
        return Request(custom_id=slot_id, params=params), spec

    def submit_scratch(self, shapes, slot_ids, topics, seed, name):
        # topics は、位置ごとの話題の辞書(ヒントから)か、話題の一覧(そこから乱数で選ぶ)
        rng = random.Random(seed)
        requests = []
        specs = {}
        for slot_id in slot_ids:
            if isinstance(topics, dict):
                topic = topics[slot_id]
            else:
                topic = rng.choice(topics)
            request, spec = self.scratch_request(slot_id, shapes.shapes[slot_id], topic)
            requests.append(request)
            specs[slot_id] = spec
        (self.out_dir / f"{name}_specs.json").write_text(json.dumps(specs, ensure_ascii=False, indent=1), encoding="utf-8")
        batch = self.client.messages.batches.create(requests=requests)
        record = {"batch_id": batch.id, "name": name, "count": len(requests)}
        (self.out_dir / f"batch_{name}.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
        logger.info("バッチを出した: %s(%d 本)", batch.id, len(requests))
        return batch.id
