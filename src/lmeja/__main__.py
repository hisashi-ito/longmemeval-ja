"""使い方
    uv run python -m lmeja trial-submit     試しの 20 問をバッチに出す
    uv run python -m lmeja trial-fetch      バッチの終わりを待ち、結果を out/trial/ に書き出して形を確かめる
    uv run python -m lmeja full-submit      試しで作った問題を除いた残りを、バッチに出す
    uv run python -m lmeja full-fetch       残りのバッチの終わりを待ち、結果を out/full/ に書き出して形を確かめる
    uv run python -m lmeja redo-submit      形の確かめで落ちた問題と、試しの 20 問を、いまの指示文で作り直す
    uv run python -m lmeja redo-fetch       作り直しの結果を out/redo/ に書き出して形を確かめる
    uv run python -m lmeja filler-trial-submit   挟まる会話を 50 本だけ訳すバッチを出す
    uv run python -m lmeja filler-trial-fetch    その結果を out/filler_trial/ に書き出して形を確かめる
    uv run python -m lmeja retry <effort> <番号...>   指定した問題だけを作り直す(out/retry/)
    uv run python -m lmeja filler-full      挟まる会話(著者の会話と UltraChat)を全部作り直す(out/filler/)
    uv run python -m lmeja scratch-full     ShareGPT の全部の位置に、話題を取り出して日本語の会話を書かせる(out/scratch/)
    uv run python -m lmeja assemble-oracle  oracle 版の 500 問を out/longmemeval_ja_oracle.json にまとめる
    uv run python -m lmeja filler-retry     挟まる会話のうち、発言の数が違ったものと読めなかったものを作り直す(out/filler_retry/)
    uv run python -m lmeja assemble-s       S 版を out/longmemeval_ja_s.json に組み立てる
    uv run python -m lmeja verify-full      500 問すべてで、英語の元と日本語の作り直しの両方を解かせて判定する
    uv run python -m lmeja verify-trial     試しの 20 問で、英語の元と日本語の作り直しの両方を解かせて判定する
    uv run python -m lmeja scratch-trial    ShareGPT の 20 か所から話題だけを取り出し、その話題で日本語の会話を書かせる
"""
import json
import sys
from pathlib import Path

from lmeja.check import ShapeChecker
from lmeja.haystack import FillerChecker, FillerLocalizer, FillerPool
from lmeja.scratch import HintMaker, ScratchMaker, SlotShapes
from lmeja.verify import Verifier
from lmeja.safety import SafetyChecker
from lmeja.localize import ROOT, ChunkedRun, Dataset, Localizer, load_conf


class Main:
    def __init__(self):
        self.params = load_conf()

    def trial_submit(self):
        dataset = Dataset(self.params)
        chosen = dataset.select_trial()
        ids = []
        for item in chosen:
            ids.append(item["question_id"])
        out_dir = ROOT / self.params["common"]["out_dir"]
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "trial_ids.json").write_text(json.dumps(ids, indent=1), encoding="utf-8")
        Localizer(self.params).submit(chosen, "trial")

    def trial_fetch(self):
        out_dir = ROOT / self.params["common"]["out_dir"]
        record = json.loads((out_dir / "batch_trial.json").read_text(encoding="utf-8"))
        localizer = Localizer(self.params)
        localizer.wait(record["batch_id"])
        localizer.fetch(record["batch_id"], "trial")
        ShapeChecker(Dataset(self.params).items).run(out_dir / "trial")

    def full_submit(self):
        out_dir = ROOT / self.params["common"]["out_dir"]
        done = set(json.loads((out_dir / "trial_ids.json").read_text(encoding="utf-8")))
        rest = []
        for item in Dataset(self.params).items:
            if item["question_id"] not in done:
                rest.append(item)
        Localizer(self.params).submit(rest, "full")

    def full_fetch(self):
        out_dir = ROOT / self.params["common"]["out_dir"]
        record = json.loads((out_dir / "batch_full.json").read_text(encoding="utf-8"))
        localizer = Localizer(self.params)
        localizer.wait(record["batch_id"])
        localizer.fetch(record["batch_id"], "full")
        ShapeChecker(Dataset(self.params).items).run(out_dir / "full")

    def redo_submit(self):
        out_dir = ROOT / self.params["common"]["out_dir"]
        targets = set(json.loads((out_dir / "trial_ids.json").read_text(encoding="utf-8")))
        shape = json.loads((out_dir / "full" / "_shape.json").read_text(encoding="utf-8"))
        for qid, problems in shape.items():
            if problems:
                targets.add(qid)
        record = json.loads((out_dir / "batch_full.json").read_text(encoding="utf-8"))
        # JSON として読めなかったものは _shape.json に載らないので、raw があって json が無いものを足す
        for raw in (out_dir / "full").glob("*.raw.txt"):
            qid = raw.name[:-len(".raw.txt")]
            if not (out_dir / "full" / f"{qid}.json").exists():
                targets.add(qid)
        items = []
        for item in Dataset(self.params).items:
            if item["question_id"] in targets:
                items.append(item)
        (out_dir / "redo_ids.json").write_text(json.dumps(sorted(targets), indent=1), encoding="utf-8")
        Localizer(self.params).submit(items, "redo")

    def redo_fetch(self):
        out_dir = ROOT / self.params["common"]["out_dir"]
        record = json.loads((out_dir / "batch_redo.json").read_text(encoding="utf-8"))
        localizer = Localizer(self.params)
        localizer.wait(record["batch_id"])
        localizer.fetch(record["batch_id"], "redo")
        ShapeChecker(Dataset(self.params).items).run(out_dir / "redo")

    def filler_trial_submit(self):
        conf = self.params["haystack"]
        pool = FillerPool(self.params)
        ids = pool.sample(conf["trial_count"], conf["seed"])
        out_dir = ROOT / self.params["common"]["out_dir"]
        (out_dir / "filler_trial_ids.json").write_text(json.dumps(ids, indent=1), encoding="utf-8")
        FillerLocalizer(self.params).submit_fillers(pool, ids, "filler_trial")

    def filler_trial_fetch(self):
        out_dir = ROOT / self.params["common"]["out_dir"]
        record = json.loads((out_dir / "batch_filler_trial.json").read_text(encoding="utf-8"))
        localizer = FillerLocalizer(self.params)
        localizer.wait(record["batch_id"])
        localizer.fetch(record["batch_id"], "filler_trial")
        FillerChecker(FillerPool(self.params)).run(out_dir / "filler_trial")

    def sharegpt_sessions(self, slot_ids):
        # 話題を取り出すためだけに、選んだ位置の ShareGPT の会話を読み込む(書き出さない)
        source = ROOT / self.params["haystack"]["source"]
        wanted = set(slot_ids)
        found = {}
        for item in json.loads(source.read_text(encoding="utf-8")):
            for sid, turns in zip(item["haystack_session_ids"], item["haystack_sessions"]):
                if sid in wanted and sid not in found:
                    found[sid] = turns
        return found

    def scratch_trial(self):
        import random
        conf = self.params["scratch"]
        out_dir = ROOT / self.params["common"]["out_dir"]
        shapes = SlotShapes(self.params)
        slot_ids = random.Random(conf["seed"]).sample(sorted(shapes.shapes), conf["trial_count"])
        (out_dir / "scratch_trial_ids.json").write_text(json.dumps(slot_ids, indent=1), encoding="utf-8")
        # 1. ShareGPT の会話から、一般化した話題だけを取り出す
        hinter = HintMaker(self.params)
        hint_batch = hinter.submit_hints(self.sharegpt_sessions(slot_ids), slot_ids, "hint_trial")
        hinter.wait(hint_batch)
        hinter.fetch(hint_batch, "hint_trial")
        topics = {}
        for slot_id in slot_ids:
            path = out_dir / "hint_trial" / (slot_id + ".json")
            if path.exists():
                topics[slot_id] = json.loads(path.read_text(encoding="utf-8"))
        # 2. 話題だけを見て、日本語の会話をゼロから書かせる
        maker = ScratchMaker(self.params)
        batch = maker.submit_scratch(shapes, list(topics), topics, conf["seed"], "scratch_trial")
        maker.wait(batch)
        maker.fetch(batch, "scratch_trial")
    def retry(self, ids, effort, model=None, name="retry", note=""):
        # 指定した問題だけを、考える深さ(とモデル)を変えて作り直す(out/<name>/ に書く)
        self.params["localize"]["effort"] = effort
        if model is not None:
            self.params["localize"]["model"] = model
        items = []
        for item in Dataset(self.params).items:
            if item["question_id"] in ids:
                items.append(item)
        localizer = Localizer(self.params)
        if note:
            localizer.instruction = localizer.instruction + "\n" + note
        batch = localizer.submit(items, name)
        localizer.wait(batch)
        localizer.fetch(batch, name)
        ShapeChecker(Dataset(self.params).items).run(ROOT / self.params["common"]["out_dir"] / name)

    def filler_full(self):
        # 著者の会話と UltraChat の会話を、試しの 50 本を除いて全部作り直す
        out_dir = ROOT / self.params["common"]["out_dir"]
        done = set(json.loads((out_dir / "filler_trial_ids.json").read_text(encoding="utf-8")))
        pool = FillerPool(self.params)
        localizer = FillerLocalizer(self.params)
        requests = []
        # すでに JSON として読めたものは飛ばす(途中で止めたあとの出し直しのため)
        for sid in sorted(pool.sessions):
            if sid not in done and not (out_dir / "filler" / (sid + ".json")).exists():
                requests.append(localizer.filler_request(sid, pool.sessions[sid]))
        print(f"挟まる会話の残り: {len(requests)} 本")
        ChunkedRun(localizer, self.params["haystack"]["chunk"]).run(requests, "filler")
        FillerChecker(pool).run(out_dir / "filler")

    def scratch_full(self):
        # ShareGPT の全部の位置について、話題を取り出し(試しで取り出した分は使い回す)、会話を書かせる
        conf = self.params["scratch"]
        out_dir = ROOT / self.params["common"]["out_dir"]
        shapes = SlotShapes(self.params)
        slot_ids = sorted(shapes.shapes)
        topics = {}
        for slot_id in slot_ids:
            path = out_dir / "hint_trial" / (slot_id + ".json")
            if path.exists():
                topics[slot_id] = json.loads(path.read_text(encoding="utf-8"))
        for slot_id in slot_ids:
            path = out_dir / "hint" / (slot_id + ".json")
            if slot_id not in topics and path.exists():
                topics[slot_id] = json.loads(path.read_text(encoding="utf-8"))
        rest = []
        for slot_id in slot_ids:
            if slot_id not in topics:
                rest.append(slot_id)
        hinter = HintMaker(self.params)
        sessions = self.sharegpt_sessions(rest)
        hint_requests = []
        for slot_id in rest:
            hint_requests.append(hinter.hint_request(slot_id, sessions[slot_id]))
        if hint_requests:
            ChunkedRun(hinter, self.params["haystack"]["chunk"]).run(hint_requests, "hint")
        for slot_id in rest:
            path = out_dir / "hint" / (slot_id + ".json")
            if path.exists():
                topics[slot_id] = json.loads(path.read_text(encoding="utf-8"))
        maker = ScratchMaker(self.params)
        requests = []
        specs = {}
        for slot_id in slot_ids:
            if slot_id in topics:
                request, spec = maker.scratch_request(slot_id, shapes.shapes[slot_id], topics[slot_id])
                specs[slot_id] = spec
                # すでに書けたものは飛ばす
                if not (out_dir / "scratch" / (slot_id + ".json")).exists():
                    requests.append(request)
        print(f"新しい会話の残り: {len(requests)} 本")
        (out_dir / "scratch_specs.json").write_text(json.dumps(specs, ensure_ascii=False, indent=1), encoding="utf-8")
        ChunkedRun(maker, self.params["haystack"]["chunk"]).run(requests, "scratch")
        # 発言の数が指定と違うもの、読めなかったものを、1 回だけ作り直す
        again = []
        for slot_id in specs:
            path = out_dir / "scratch" / (slot_id + ".json")
            if not path.exists():
                again.append(slot_id)
                continue
            made = json.loads(path.read_text(encoding="utf-8"))
            if len(made.get("turns", [])) != specs[slot_id]["turns"]:
                again.append(slot_id)
        logger_line = f"発言の数が違うか読めなかったもの: {len(again)} 本を作り直す"
        print(logger_line)
        retry_requests = []
        for slot_id in again:
            request, spec = maker.scratch_request(slot_id, shapes.shapes[slot_id], topics[slot_id])
            retry_requests.append(request)
        if retry_requests:
            ChunkedRun(maker, self.params["haystack"]["chunk"]).run(retry_requests, "scratch_retry")

    def assemble_oracle(self):
        # 試し、本番、作り直しに散らばった結果から、問題ごとにいちばん新しいものを採り、元と同じ形の 1 ファイルにする
        out_dir = ROOT / self.params["common"]["out_dir"]
        order = ["repair2", "repair", "retry", "redo", "full"]
        made = {}
        for name in order:
            for path in sorted((out_dir / name).glob("*.json")):
                if path.name.startswith("_") or path.name.endswith(".error.json"):
                    continue
                item = json.loads(path.read_text(encoding="utf-8"))
                if item["question_id"] not in made:
                    made[item["question_id"]] = item
        checker = ShapeChecker(Dataset(self.params).items)
        rows = []
        missing = []
        for original in Dataset(self.params).items:
            qid = original["question_id"]
            item = made.get(qid)
            if item is None or checker.check(item):
                missing.append(qid)
                continue
            sessions = []
            for session in item["sessions"]:
                turns = []
                for turn in session["turns"]:
                    # 元の LongMemEval と同じく、すべての発言に has_answer を true か false で付ける
                    turns.append({"role": turn["role"], "content": turn["content"],
                                  "has_answer": bool(turn.get("has_answer"))})
                sessions.append(turns)
            rows.append({
                "question_id": qid,
                "question_type": original["question_type"],
                "question": item["question"],
                # 答えは文字にそろえる(元は 32 問が整数で、Hugging Face の表示が型の混在で止まっていた)
                "answer": str(item["answer"]),
                "question_date": original["question_date"],
                "haystack_session_ids": original["haystack_session_ids"],
                "haystack_dates": original["haystack_dates"],
                "haystack_sessions": sessions,
                "answer_session_ids": original["answer_session_ids"],
            })
        path = out_dir / "longmemeval_ja_oracle.json"
        path.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"oracle 版: {len(rows)} 問を書いた。欠け {len(missing)} 問 {missing}")
        return rows

    @staticmethod
    def as_made(row):
        # 確かめの台本が読む形(sessions に日付と発言)に直す
        sessions = []
        for date, turns in zip(row["haystack_dates"], row["haystack_sessions"]):
            sessions.append({"date": date, "turns": turns})
        return {"question_id": row["question_id"], "question": row["question"], "answer": str(row["answer"]),
                "question_date": row["question_date"], "sessions": sessions}

    def verify_trial(self):
        out_dir = ROOT / self.params["common"]["out_dir"]
        ja_rows = self.assemble_oracle()
        ids = set(json.loads((out_dir / "trial_ids.json").read_text(encoding="utf-8")))
        types = {}
        ja = []
        for row in ja_rows:
            if row["question_id"] in ids:
                ja.append(self.as_made(row))
                types[row["question_id"]] = row["question_type"]
        en = []
        for row in Dataset(self.params).items:
            if row["question_id"] in ids:
                en.append(self.as_made(row))
        verifier = Verifier(self.params)
        verifier.run(ja, types, "verify_trial_ja")
        verifier.run(en, types, "verify_trial_en")

    def verify_full(self):
        # 500 問すべてを、日本語の作り直しと英語の元の両方で解かせて判定する(1 本 500 件ずつ、順に)
        ja_rows = self.assemble_oracle()
        types = {}
        ja = []
        for row in ja_rows:
            ja.append(self.as_made(row))
            types[row["question_id"]] = row["question_type"]
        en = []
        for row in Dataset(self.params).items:
            en.append(self.as_made(row))
        verifier = Verifier(self.params)
        verifier.run(ja, types, "verify_full_ja")
        verifier.run(en, types, "verify_full_en")

    def assemble_s(self):
        # 元の S 版の骨組み(会話の並び、日付、根拠の位置)のまま、中身を日本語に差し替えて S 版を作る。
        # 根拠の会話は oracle 版の作り直し、著者の会話と UltraChat は挟まる会話の作り直し、ShareGPT の位置は
        # 話題から新しく書いた会話。まだ無い会話がある問題は書かずに、数だけ報告する
        out_dir = ROOT / self.params["common"]["out_dir"]
        oracle = {}
        for row in json.loads((out_dir / "longmemeval_ja_oracle.json").read_text(encoding="utf-8")):
            for sid, turns in zip(row["haystack_session_ids"], row["haystack_sessions"]):
                oracle[sid] = turns
        source = json.loads((ROOT / self.params["haystack"]["source"]).read_text(encoding="utf-8"))
        ja_oracle = {}
        for row in json.loads((out_dir / "longmemeval_ja_oracle.json").read_text(encoding="utf-8")):
            ja_oracle[row["question_id"]] = row
        rows = []
        missing = {}
        for item in source:
            qid = item["question_id"]
            evidence = set(item["answer_session_ids"])
            sessions = []
            lacking = 0
            for sid, turns in zip(item["haystack_session_ids"], item["haystack_sessions"]):
                if sid in oracle:
                    made = oracle[sid]
                elif sid.startswith("sharegpt"):
                    made = self.load_turns(out_dir / "scratch_fix" / (sid + ".json"),
                                           out_dir / "scratch_replace" / (sid + ".json"),
                                           out_dir / "scratch_retry" / (sid + ".json"),
                                           out_dir / "scratch" / (sid + ".json"),
                                           out_dir / "scratch_trial" / (sid + ".json"))
                else:
                    # 新しい作り直しから順に見て、発言の数が元と同じものを使う(違うものは根拠の印の位置がずれる)
                    made = None
                    for layer in ("filler_fix", "filler_retry2", "filler_retry", "filler", "filler_trial"):
                        candidate = self.load_turns(out_dir / layer / (sid + ".json"))
                        if candidate is not None and len(candidate) == len(turns):
                            made = candidate
                            break
                if made is None:
                    lacking += 1
                    sessions.append(None)
                    continue
                marked = []
                for turn in made:
                    marked.append({"role": turn["role"], "content": turn["content"],
                                   "has_answer": bool(turn.get("has_answer")) and sid in evidence})
                sessions.append(marked)
            if lacking > 0 or qid not in ja_oracle:
                missing[qid] = lacking
                continue
            base = ja_oracle[qid]
            rows.append({
                "question_id": qid,
                "question_type": item["question_type"],
                "question": base["question"],
                "answer": base["answer"],
                "question_date": item["question_date"],
                "haystack_session_ids": item["haystack_session_ids"],
                "haystack_dates": item["haystack_dates"],
                "haystack_sessions": sessions,
                "answer_session_ids": item["answer_session_ids"],
            })
        path = out_dir / "longmemeval_ja_s.json"
        path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        print(f"S 版: {len(rows)} 問を書いた。会話が足りない問題 {len(missing)} 問(足りない会話は合わせて "
              f"{sum(missing.values())} 本)")
        return rows, missing

    @staticmethod
    def load_turns(*paths):
        # 最初に見つかったファイルの発言を返す。発言の数が指定と違う新しい会話は、作り直し(scratch_retry)を先に見る
        for path in paths:
            if path.exists():
                made = json.loads(path.read_text(encoding="utf-8"))
                turns = made.get("turns")
                if isinstance(turns, list) and len(turns) > 0:
                    ok = True
                    for turn in turns:
                        if not isinstance(turn, dict) or not isinstance(turn.get("content"), str):
                            ok = False
                    if ok:
                        return turns
        return None

    def filler_retry(self, name="filler_retry", model=None):
        # 挟まる会話のうち、発言の数が元と違ったものと、まだ読めていないものを、発言の数を念押しして作り直す。
        # name を変えて呼ぶと、前の作り直しでも直らなかったものだけを対象にする(model で強いモデルに替える)
        out_dir = ROOT / self.params["common"]["out_dir"]
        pool = FillerPool(self.params)
        trial = set(json.loads((out_dir / "filler_trial_ids.json").read_text(encoding="utf-8")))
        layers = ["filler", "filler_retry", "filler_retry2"]
        shapes = {}
        for layer in layers:
            path = out_dir / layer / "_shape.json"
            shapes[layer] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        targets = []
        for sid in sorted(pool.sessions):
            if sid in trial:
                continue
            good = False
            for layer in layers:
                if (out_dir / layer / (sid + ".json")).exists() and not shapes[layer].get(sid, ["?"]):
                    good = True
            if not good:
                targets.append(sid)
        if model is not None:
            self.params["haystack"]["model"] = model
        localizer = FillerLocalizer(self.params)
        requests = []
        for sid in targets:
            note = ("\n\nこの会話は発言が " + str(len(pool.sessions[sid])) + " 個ある。出力の turns も必ず "
                    + str(len(pool.sessions[sid])) + " 個にし、元の発言と 1 対 1 で対応させる。発言をまとめたり分けたりしない。")
            request = localizer.filler_request(sid, pool.sessions[sid])
            request["params"]["system"] = [{"type": "text", "text": localizer.instruction + note}]
            requests.append(request)
        print(f"挟まる会話の作り直し({name}): {len(requests)} 本")
        ChunkedRun(localizer, self.params["haystack"]["chunk"]).run(requests, name)
        FillerChecker(pool).run(out_dir / name)

    def rejudge(self):
        # 英語と日本語で判定が分かれた問題だけを、強いモデル(conf の verify.rejudge_model)で判定し直す
        # (依頼者 2026-11-18「どうしても難しいやつは Fable 5 に任せてよい。判断するやつの話」)
        out_dir = ROOT / self.params["common"]["out_dir"]
        ja = json.loads((out_dir / "verify_full_ja.json").read_text(encoding="utf-8"))
        en = json.loads((out_dir / "verify_full_en.json").read_text(encoding="utf-8"))
        verifier = Verifier(self.params)
        verifier.judge_model = self.params["verify"]["rejudge_model"]
        requests = []
        for qid in ja["report"]:
            a = ja["report"][qid]
            b = en["report"][qid]
            if a["passed"] == b["passed"]:
                continue
            for lang, row in (("ja", a), ("en", b)):
                made = {"question_id": lang + "__" + qid, "question": row["question"], "answer": row["answer"]}
                request = verifier.judge_request(made, row["type"], row["hypothesis"])
                if qid.endswith("_abs"):
                    request = verifier.judge_request({"question_id": lang + "__" + qid + "_abs_",
                                                      "question": row["question"], "answer": row["answer"]},
                                                     "abstention", row["hypothesis"])
                    request["custom_id"] = lang + "__" + qid
                requests.append(request)
        batch = verifier.submit_requests(requests, "rejudge")
        verifier.wait(batch)
        verdicts, usage = verifier.texts(batch)
        result = {}
        for key, text in verdicts.items():
            lang, qid = key.split("__", 1)
            result.setdefault(qid, {})[lang] = {"verdict": text, "passed": verifier.passed(text)}
        (out_dir / "rejudge.json").write_text(json.dumps({"usage": usage, "result": result}, ensure_ascii=False,
                                                         indent=1), encoding="utf-8")
        both = 0
        en_only = []
        for qid, row in result.items():
            if row.get("ja", {}).get("passed") and row.get("en", {}).get("passed"):
                both += 1
            elif row.get("en", {}).get("passed") and not row.get("ja", {}).get("passed"):
                en_only.append(qid)
        print(f"判定し直し: {len(result)} 問のうち、両方とも正解 {both} 問、英語だけ正解 {len(en_only)} 問 {en_only}")

    def scratch_retry(self, model=None):
        # 新しい会話のうち、無いものと、発言の一覧の形が違うものだけを書き直す。発言の数は問わない
        # (新しい会話には根拠の印が無く、元の ShareGPT の中身とも対応しないので、数をそろえる意味がない)
        out_dir = ROOT / self.params["common"]["out_dir"]
        specs = json.loads((out_dir / "scratch_specs.json").read_text(encoding="utf-8"))
        shapes = SlotShapes(self.params)
        topics = {}
        for folder in ("hint", "hint_trial"):
            for path in (out_dir / folder).glob("*.json"):
                if not path.name.startswith("_"):
                    topics[path.stem] = json.loads(path.read_text(encoding="utf-8"))
        if model is not None:
            self.params["scratch"]["model"] = model
        maker = ScratchMaker(self.params)
        requests = []
        for slot_id in specs:
            turns = self.load_turns(out_dir / "scratch_retry" / (slot_id + ".json"),
                                    out_dir / "scratch" / (slot_id + ".json"),
                                    out_dir / "scratch_trial" / (slot_id + ".json"))
            if turns is not None or slot_id not in topics:
                continue
            request, spec = maker.scratch_request(slot_id, shapes.shapes[slot_id], topics[slot_id])
            requests.append(request)
        print(f"新しい会話の書き直し: {len(requests)} 本")
        if requests:
            ChunkedRun(maker, self.params["haystack"]["chunk"]).run(requests, "scratch_retry")

    def verify_ids(self, ids):
        # 作り直した問題だけを、日本語で解かせ、強いモデル(verify.rejudge_model)で判定する
        ja_rows = self.assemble_oracle()
        types = {}
        ja = []
        for row in ja_rows:
            if row["question_id"] in ids:
                ja.append(self.as_made(row))
                types[row["question_id"]] = row["question_type"]
        verifier = Verifier(self.params)
        verifier.judge_model = self.params["verify"]["rejudge_model"]
        verifier.run(ja, types, "verify_repair_ja")

    def safety(self):
        # ShareGPT の位置に書いた会話を 1 本ずつ判定する(out/safety/ に判定、out/safety_unsafe.json に有害の一覧)
        out_dir = ROOT / self.params["common"]["out_dir"]
        specs = json.loads((out_dir / "scratch_specs.json").read_text(encoding="utf-8"))
        checker = SafetyChecker(self.params)
        requests = []
        for slot_id in specs:
            if (out_dir / "safety" / (slot_id + ".json")).exists():
                continue
            turns = self.load_turns(out_dir / "scratch_retry" / (slot_id + ".json"),
                                    out_dir / "scratch" / (slot_id + ".json"),
                                    out_dir / "scratch_trial" / (slot_id + ".json"))
            if turns is not None:
                requests.append(checker.request(slot_id, turns))
        print(f"有害の判定: {len(requests)} 本")
        if requests:
            ChunkedRun(checker, self.params["haystack"]["chunk"]).run(requests, "safety")
        unsafe = {}
        for path in (out_dir / "safety").glob("*.json"):
            if path.name.startswith("_"):
                continue
            row = json.loads(path.read_text(encoding="utf-8"))
            if row.get("unsafe"):
                unsafe[path.stem] = row.get("reason", "")
        (out_dir / "safety_unsafe.json").write_text(json.dumps(unsafe, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"有害と判定: {len(unsafe)} 本")

    def replace_unsafe(self):
        # 有害と判定された位置と、元の依頼がはっきり有害だった位置を、ほかの位置の無害な話題に替えて書き直す
        # (out/scratch_replace/。置き換えの対応は out/topic_replacements.json)。書き直したものも判定に通す
        import random
        out_dir = ROOT / self.params["common"]["out_dir"]
        unsafe = set(json.loads((out_dir / "safety_unsafe.json").read_text(encoding="utf-8")))
        listed = set(self.params["safety"]["replace_slots"])
        targets = sorted(unsafe | listed)
        specs = json.loads((out_dir / "scratch_specs.json").read_text(encoding="utf-8"))
        topics = {}
        for folder in ("hint", "hint_trial"):
            for path in (out_dir / folder).glob("*.json"):
                if not path.name.startswith("_"):
                    topics[path.stem] = json.loads(path.read_text(encoding="utf-8"))
        safe_pool = []
        for slot_id in sorted(topics):
            if slot_id not in unsafe and slot_id not in listed and slot_id in specs:
                safe_pool.append(slot_id)
        rng = random.Random(self.params["scratch"]["seed"])
        shapes = SlotShapes(self.params)
        maker = ScratchMaker(self.params)
        # 話題を取り出す段で失敗して話題が無い位置も、同じく別の位置の話題を当てる
        for slot_id in sorted(shapes.shapes):
            if slot_id not in topics and slot_id not in targets:
                targets.append(slot_id)
        mapping = {}
        path = out_dir / "topic_replacements.json"
        if path.exists():
            mapping = json.loads(path.read_text(encoding="utf-8"))
        # すでに入れ替えに使った話題は、もう一度は使わない(同じ話題の会話が増えないように)
        used = set()
        for value in mapping.values():
            used.add(value["topic_from"])
        remaining = []
        for slot_id in safe_pool:
            if slot_id not in used:
                remaining.append(slot_id)
        safe_pool = remaining
        requests = []
        for slot_id in targets:
            if slot_id not in shapes.shapes:
                continue
            if (out_dir / "scratch_replace" / (slot_id + ".json")).exists():
                continue
            donor = rng.choice(safe_pool)
            mapping[slot_id] = {"topic_from": donor, "topic": topics[donor]["topic"]}
            request, spec = maker.scratch_request(slot_id, shapes.shapes[slot_id], topics[donor])
            requests.append(request)
        path.write_text(json.dumps(mapping, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"話題を替えて書き直す: {len(requests)} 本")
        ChunkedRun(maker, self.params["haystack"]["chunk"]).run(requests, "scratch_replace")
        checker = SafetyChecker(self.params)
        checks = []
        for slot_id in mapping:
            turns = self.load_turns(out_dir / "scratch_replace" / (slot_id + ".json"))
            if turns is not None:
                checks.append(checker.request(slot_id, turns))
        ChunkedRun(checker, self.params["haystack"]["chunk"]).run(checks, "safety_replace")
        bad = []
        for slot_id in mapping:
            path = out_dir / "safety_replace" / (slot_id + ".json")
            if not path.exists() or json.loads(path.read_text(encoding="utf-8")).get("unsafe"):
                bad.append(slot_id)
        print(f"書き直しの判定: {len(mapping)} 本のうち、まだ有害か判定できなかったもの {len(bad)} 本 {bad}")


    def fix_english(self):
        # S 版の形式の検査で、英語のまま残った発言があった会話(out/english_left.json)を Sonnet で作り直す。
        # 挟まる会話は filler_fix/ へ、新しい会話は scratch_fix/ へ書き、組み立てではこれを最優先で使う
        out_dir = ROOT / self.params["common"]["out_dir"]
        left = json.loads((out_dir / "english_left.json").read_text(encoding="utf-8"))
        pool = FillerPool(self.params)
        self.params["haystack"]["model"] = "claude-sonnet-5-5"
        self.params["scratch"]["model"] = "claude-sonnet-5-5"
        localizer = FillerLocalizer(self.params)
        maker = ScratchMaker(self.params)
        shapes = SlotShapes(self.params)
        topics = {}
        for folder in ("hint", "hint_trial"):
            for path in (out_dir / folder).glob("*.json"):
                if not path.name.startswith("_"):
                    topics[path.stem] = json.loads(path.read_text(encoding="utf-8"))
        replaced = {}
        path = out_dir / "topic_replacements.json"
        if path.exists():
            replaced = json.loads(path.read_text(encoding="utf-8"))
        fillers = []
        scratches = []
        for sid in sorted(left):
            if sid.startswith("sharegpt"):
                topic = topics[sid]
                if sid in replaced:
                    topic = topics[replaced[sid]["topic_from"]]
                request, spec = maker.scratch_request(sid, shapes.shapes[sid], topic)
                note = "\n\n発言はすべて日本語で書く。英語の文を残さない(作品名、商品名、コードは除く)。"
                request["params"]["system"][0]["text"] = request["params"]["system"][0]["text"] + note
                scratches.append(request)
            elif sid in pool.sessions:
                note = ("\n\nこの会話は発言が " + str(len(pool.sessions[sid])) + " 個ある。出力の turns も必ず "
                        + str(len(pool.sessions[sid])) + " 個にし、元の発言と 1 対 1 で対応させる。"
                        + "すべての発言を日本語にする。英語のまま残さない(作品名、商品名、コードは除く)。")
                request = localizer.filler_request(sid, pool.sessions[sid])
                request["params"]["system"] = [{"type": "text", "text": localizer.instruction + note}]
                fillers.append(request)
        print(f"英語が残った会話の作り直し: 挟まる会話 {len(fillers)} 本、新しい会話 {len(scratches)} 本")
        if fillers:
            ChunkedRun(localizer, self.params["haystack"]["chunk"]).run(fillers, "filler_fix")
        if scratches:
            ChunkedRun(maker, self.params["haystack"]["chunk"]).run(scratches, "scratch_fix")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    main = Main()
    if command == "trial-submit":
        main.trial_submit()
    elif command == "trial-fetch":
        main.trial_fetch()
    elif command == "full-submit":
        main.full_submit()
    elif command == "full-fetch":
        main.full_fetch()
    elif command == "redo-submit":
        main.redo_submit()
    elif command == "redo-fetch":
        main.redo_fetch()
    elif command == "filler-trial-submit":
        main.filler_trial_submit()
    elif command == "filler-trial-fetch":
        main.filler_trial_fetch()
    elif command == "retry":
        main.retry(set(sys.argv[3:]), sys.argv[2])
    elif command == "filler-full":
        main.filler_full()
    elif command == "scratch-full":
        main.scratch_full()
    elif command == "assemble-oracle":
        main.assemble_oracle()
    elif command == "verify-trial":
        main.verify_trial()
    elif command == "filler-retry":
        main.filler_retry()
    elif command == "filler-retry2":
        main.filler_retry("filler_retry2", "claude-sonnet-5-5")
    elif command == "assemble-s":
        main.assemble_s()
    elif command == "repair":
        # 答えの確かめで壊れていると分かった問題を、強いモデルで念押しして作り直す(out/repair/)
        note = (Path(__file__).parent / "repair_note.md").read_text(encoding="utf-8")
        main.retry(set(sys.argv[2:]), "high", "claude-opus-5-5", "repair", note)
    elif command == "verify-ids":
        main.verify_ids(set(sys.argv[2:]))
    elif command == "repair-sonnet":
        # Opus が答えを控えて止まった問題を、Sonnet で同じ念押しを付けて作り直す(out/repair2/)
        note = (Path(__file__).parent / "repair_note.md").read_text(encoding="utf-8")
        main.retry(set(sys.argv[2:]), "high", "claude-sonnet-5-5", "repair2", note)
    elif command == "scratch-retry":
        main.scratch_retry(sys.argv[2] if len(sys.argv) > 2 else None)
    elif command == "fix-english":
        main.fix_english()
    elif command == "replace-unsafe":
        main.replace_unsafe()
    elif command == "safety":
        main.safety()
    elif command == "rejudge":
        main.rejudge()
    elif command == "verify-full":
        main.verify_full()
    elif command == "scratch-trial":
        main.scratch_trial()
    else:
        print(__doc__)
