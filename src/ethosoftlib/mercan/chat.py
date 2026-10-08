"""Mercan SFT's ChatML formatter, separated from the generic Ethosoft SDK core."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

DEFAULT_SYSTEM_PROMPT = (
    "Sen Mercan'sın, MercanAI tarafından Türkçe konuşan kullanıcılar için geliştirilmiş "
    "bir yapay zeka asistanısın. Sorulara açık, doğru ve anlaşılır bir Türkçeyle yanıt "
    "verirsin; emin olmadığın konularda bunu açıkça belirtir, tahmini bilgiyi gerçek "
    "gibi sunmazsın. Kısa sorulara kısa ve net, ayrıntı gerektiren sorulara ise "
    "gerektiği kadar açıklayıcı cevaplar verirsin. Kullanıcıya karşı saygılı, sabırlı "
    "ve yardımsever bir tutum sergilersin; zararlı, yasa dışı veya güvenlik açısından "
    "tehlikeli isteklerde bulunulduğunda kibarca reddeder, bunun yerine güvenli ve "
    "yapıcı bir alternatif sunmaya çalışırsın."
)

ROLE_NAMES = {"system": "sistem", "user": "kullanici", "assistant": "asistan"}


def format_chat(messages: Sequence[Mapping[str, str]]) -> str:
    """Render turns using the same structural control tokens as Mercan CLI.

    The caller supplies trusted role names; contents are escaped to avoid
    accidentally treating literal user text as ChatML delimiters.
    """
    out: list[str] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role not in ROLE_NAMES or not isinstance(content, str):
            raise ValueError("Messages must contain a known role and string content")
        safe_content = content.replace("<|", "<​|")
        out.append(f"<|im_start|>{ROLE_NAMES[role]}\n{safe_content}<|im_end|>\n")
    out.append("<|im_start|>asistan\n")
    return "".join(out)
