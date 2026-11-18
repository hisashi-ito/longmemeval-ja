"""ShareGPT の位置に書いた新しい会話に、有害な中身が入っていないかを判定する。

ShareGPT は実在の利用者の会話で、違法な依頼や AI の制限を外す試みも混ざる。話題だけを取り出して書いた会話にも、
その中身が残ることがある(偽造の身分証明書の広告文を書き始めた例があった)。公開するデータに入れないよう、
1 本ずつ判定し、有害なものは話題を替えて書き直す。
"""
import json
import logging

from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from lmeja.localize import Localizer

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s -- : %(message)s')
logger = logging.getLogger(__name__)

SAFETY_INSTRUCTION = """次は、公開するベンチマークのデータに入れる、利用者と AI アシスタントの日本語の会話です。
次のどれかを含むかを判定してください。
- 違法な行為の手助け(偽造、詐欺、不正アクセス、他人の情報を盗む、薬物の入手や製造、武器や爆発物の製造)
- 性的な内容(とくに未成年、実在の人物)、性暴力の容認
- 自傷や自殺の手助け
- AI の制限を外させる試み(脱獄)や、それに従った出力
- 実在の個人への中傷や、個人を特定できる情報
- 差別や憎悪をあおる内容
防犯、詐欺の見分け方、ニュースや歴史の説明、創作の中の穏当な悪役、ゲームの武器、医療の一般的な説明は、含むと判定しない。

次の形の JSON だけを出力する。前置きや説明、コードの囲みは付けない。
{"unsafe": true か false, "reason": "含むと判定した理由を短く。含まなければ空"}"""


class SafetyChecker(Localizer):
    def __init__(self, params):
        super().__init__(params)
        self.model = params["safety"]["model"]

    def request(self, slot_id, turns):
        text = ""
        for turn in turns:
            text += "[" + turn["role"] + "] " + turn["content"] + "\n"
        params = MessageCreateParamsNonStreaming(
            model=self.model,
            max_tokens=1000,
            output_config={"effort": "low"},
            system=[{"type": "text", "text": SAFETY_INSTRUCTION, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": text[:12000]}],
        )
        return Request(custom_id=slot_id, params=params)
