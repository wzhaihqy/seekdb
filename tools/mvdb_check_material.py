#!/usr/bin/env python3
"""Check agent/conversation.json against the task's format rules.

The task package states the rules for the Agent prompt material:

  * one JSON object, UTF-8, no duplicate keys, no NaN or Infinity;
  * at most 128 KiB;
  * five dimensions, named exactly as the task names them;
  * every evidence entry is an RFC6901 pointer plus a quote of at most 240
    characters, and the quote is the text the pointer points at;
  * every pointer resolves, and every quote is used by only one instance.

This script checks those, and reports what it found.  It exits non-zero
when a rule is broken, so it can be a gate before a submission.

Usage:
    python mvdb_check_material.py [path]
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

DIMENSIONS = (
    "问题理解与自主拆解",
    "技术理解与迁移",
    "独立验证与质疑",
    "实验与反馈迭代",
    "取舍与反思",
)

MAX_BYTES = 128 * 1024
MAX_QUOTE = 240


def reject_duplicates(pairs):
    """The parser that refuses a repeated key instead of keeping the last."""

    seen = {}
    for key, value in pairs:
        if key in seen:
            raise ValueError("duplicate key: %r" % key)
        seen[key] = value
    return seen


def resolve(document, pointer):
    """The value a RFC6901 pointer points at, or None."""

    if pointer == "":
        return document
    if not pointer.startswith("/"):
        return None
    current = document
    for raw in pointer[1:].split("/"):
        token = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(current, list):
            try:
                current = current[int(token)]
            except (ValueError, IndexError):
                return None
        elif isinstance(current, dict):
            if token not in current:
                return None
            current = current[token]
        else:
            return None
    return current


def _mentions_literal(text, token):
    """Whether the text holds the token as a JSON value, not as a word.

    The rules forbid NaN and Infinity as values.  A sentence that says the
    implementation rejects NaN is not a violation, so the check looks for
    the token standing on its own -- behind a quote, a colon or a bracket.
    """

    for index in range(len(text) - len(token) + 1):
        if text[index:index + len(token)] != token:
            continue
        before = text[index - 1] if index else " "
        after = text[index + len(token)] if index + len(token) < len(text) else " "
        if before in " \t\n\r:,[{" and after in " \t\n\r,]}":
            return True
    return False


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    path = argv[0] if argv else os.path.join(REPO, "agent", "conversation.json")

    problems = []
    notes = []

    raw = open(path, "rb").read()
    size = len(raw)
    if size > MAX_BYTES:
        problems.append("size %d bytes > %d KiB" % (size, MAX_BYTES))

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        problems.append("not valid UTF-8: %s" % exc)
        print("FAIL")
        return 1

    # A non-finite number is a JSON extension; the standard grammar has no
    # way to write one.  The bare words may still appear inside a sentence
    # that talks about rejecting them, so the check looks for the literals
    # rather than for the words.
    for token in ("NaN", "Infinity", "-Infinity"):
        for match in (token,):
            if _mentions_literal(text, match):
                problems.append("the document contains the literal %s" % match)

    try:
        document = json.loads(text, object_pairs_hook=reject_duplicates)
    except ValueError as exc:
        problems.append("not valid JSON: %s" % exc)
        print("FAIL")
        return 1

    if not isinstance(document, dict):
        problems.append("the top level is not an object")

    # -- the dimensions -------------------------------------------------
    dimensions = document.get("dimensions")
    if not isinstance(dimensions, list):
        problems.append("dimensions is missing or is not a list")
        dimensions = []
    names = [d.get("name") for d in dimensions if isinstance(d, dict)]
    if names != list(DIMENSIONS):
        problems.append("the dimension names are %r, expected %r"
                        % (names, list(DIMENSIONS)))

    instances = 0
    quotes = {}
    for dimension in dimensions:
        if not isinstance(dimension, dict):
            continue
        for instance in dimension.get("instances") or []:
            if not isinstance(instance, dict):
                problems.append("an instance of %r is not an object"
                                % dimension.get("name"))
                continue
            instances += 1
            name = instance.get("id", "?")

            for field in ("prompt_quote", "claim", "method", "verification",
                          "transfer", "limit"):
                value = instance.get(field)
                if not isinstance(value, str) or not value.strip():
                    problems.append("instance %s has no %s" % (name, field))

            # The two kinds of evidence the rule asks for: what the student
            # said, and what the experiment produced.  An instance that
            # carries only one of them reads as a claim without a result.
            for field in ("student_evidence", "result_evidence"):
                value = instance.get(field)
                if not isinstance(value, list) or not value:
                    problems.append("instance %s has no %s" % (name, field))

            # The response pointer has to reach a string.
            pointer = instance.get("response")
            if not isinstance(pointer, str):
                problems.append("instance %s has no response pointer" % name)
            else:
                target = resolve(document, pointer)
                if not isinstance(target, str):
                    problems.append("instance %s: the response pointer %r "
                                    "does not resolve to a string"
                                    % (name, pointer))

            prompt = instance.get("prompt")
            if isinstance(prompt, str):
                target = resolve(document, prompt)
                if not isinstance(target, str):
                    problems.append("instance %s: the prompt pointer %r "
                                    "does not resolve to a string"
                                    % (name, prompt))

            for field in ("student_evidence", "result_evidence"):
                for entry in instance.get(field) or []:
                    if not isinstance(entry, dict):
                        problems.append("instance %s: a %s entry is not an "
                                        "object" % (name, field))
                        continue
                    quote = entry.get("quote")
                    target_pointer = entry.get("pointer")
                    if not isinstance(quote, str) or \
                            not isinstance(target_pointer, str):
                        problems.append("instance %s: a malformed %s entry"
                                        % (name, field))
                        continue
                    if len(quote) > MAX_QUOTE:
                        problems.append("instance %s: a %s quote is %d "
                                        "characters (> %d)"
                                        % (name, field, len(quote), MAX_QUOTE))
                    value = resolve(document, target_pointer)
                    if not isinstance(value, str):
                        problems.append("instance %s: the %s pointer %r does "
                                        "not resolve to a string"
                                        % (name, field, target_pointer))
                    elif not (value.startswith(quote) or quote in value):
                        problems.append("instance %s: the %s quote is not the "
                                        "text at %r"
                                        % (name, field, target_pointer))
                    quotes.setdefault(quote, []).append(name)

    for quote, owners in quotes.items():
        distinct = sorted(set(owners))
        if len(distinct) > 1:
            problems.append("the quote %r is used by %d instances (%s)"
                            % (quote[:40], len(distinct), ", ".join(distinct)))

    notes.append("size: %d bytes (%.1f KiB of %d KiB)"
                 % (size, size / 1024.0, MAX_BYTES // 1024))
    notes.append("dimensions: %d, instances: %d, distinct quotes: %d"
                 % (len(dimensions), instances, len(quotes)))

    print("=== agent/conversation.json 合规检查 ===")
    for note in notes:
        print("  " + note)
    if problems:
        print()
        print("  %d problem(s):" % len(problems))
        for problem in problems:
            print("    - " + problem)
        print()
        print("FAIL")
        return 1
    print()
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
