from __future__ import annotations

import re
import unicodedata
from urllib.parse import urlparse

from src.rag.service import personalized_medical_request

from .models import EvidenceItem

# A request is personal advice when it talks about the asker (or someone they care for) AND asks what to
# take, how much, whether it is safe, or for a diagnosis. Questions about a document ("what does the
# leaflet say about...") have no first-person marker, so they pass.
PERSONAL_MARKER = re.compile(
    r"\b(mi|mis|yo|tengo|estoy|soy|peso|pesa|mi hijo|my|i|i'm|im|i've|ive|we|nuestro)\b"
    r"|\bme (tomo|puedo|vale|conviene|duele|recomiendas|sirve|doy)\b|\b(para|de) mi\b|\bse me\b"
)
ADVICE_MARKER = re.compile(
    r"\b(cuant[oa]s?|how (many|much)|recomiend\w*|recomienda\w*|recommend\w*|conviene|mejor|best|better|"
    r"prescribe\w*|prescribir\w*|receta\w*|recetame|dime|tell me|give me|dame|aconsej\w*|"
    r"diagnos\w*|que tengo|what do i have|seguro|safe|ok|okay|vale|puedo|puede|debo|deberia|should|can|could|may|"
    r"tomar|tomo|doy|darle|dar|take|give|mix|mezclar|duplicar|double|skip|saltar|saltarme|reducir|aumentar|"
    r"dejar|stop|anadir|add|combinar|combine|usar|use)\b"
    r"|\bque (dosis|cantidad|jarabe|pastilla|medicamento|antibiotico|farmaco|tomo|tomar|doy|le doy|me tomo)\b"
    r"|\bwhat (dose|can i|should i|to take|medicine|drug|antibiotic)\b"
)

# Dosing-quantity questions are refused even without a first-person marker: "how many pills a day" is
# a dosing request whoever asks it. Document phrasing ("according to the leaflet") uses other words.
STANDALONE_DOSING = re.compile(
    r"\bcuant[oa]s? (pastillas|pildoras|comprimidos|capsulas|gotas|sobres|ampollas)\b"
    r"|\bhow many (tablets|pills|capsules|drops|sachets)\b"
)

# "How many tablets does the pack contain" is a catalogue question, not a dosing request.
PACK_CONTENT = re.compile(r"(envase|caja|contiene|incluye|vienen|trae|blister|pack|package|box|contain)")

INJECTION_PATTERNS = (
    r"ignore (all |any |the )?(previous|prior|above)( instructions| rules)?",
    r"ignora (todas )?(las |tus )?(instrucciones|reglas)",
    r"ignora lo anterior",
    r"\bsystem prompt\b",
    r"olvida (tus|las|todas)( fuentes| reglas| instrucciones)",
    r"disregard (your|the|all) (guidelines|rules|instructions)",
    r"you are now (a |an )?\w+",
    r"\bpretend (you|to)\b",
    r"no restrictions|sin restricciones",
    r"repite tus instrucciones|repeat your (system )?(instructions|prompt)",
    r"(revela|muestra|show|reveal) .*(prompt|instrucciones del sistema|system instructions)",
    r"omite (las |tus )?(fuentes|citas|referencias)",
    r"(actua|act|finge) (como|as) (un |a )?(medico|doctor|physician)",
    r"without (citing|sources)|sin (citar|fuentes)",
)


def fold(text: str) -> str:
    return _normalize(text)


def _normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    plain = "".join(char for char in decomposed if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", plain)


def blocked_request(question: str) -> str | None:
    """Return a reason code when the request must be refused before any model call."""
    if personalized_medical_request(question):
        return "personalized_medical_request"
    normalized = _normalize(question)
    if any(re.search(pattern, normalized) for pattern in INJECTION_PATTERNS):
        return "prompt_injection"
    if STANDALONE_DOSING.search(normalized) and not PACK_CONTENT.search(normalized):
        return "personalized_medical_request"
    if PERSONAL_MARKER.search(normalized) and ADVICE_MARKER.search(normalized):
        return "personalized_medical_request"
    return None


def official_source(url: str, allowed_hosts: tuple[str, ...]) -> bool:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    return parsed.scheme == "https" and any(
        host == allowed or host.endswith("." + allowed) for allowed in allowed_hosts
    )


class EvidenceRegistry:
    """Evidence seen in tool results. Only registered items may appear in an answer."""

    def __init__(self, allowed_hosts: tuple[str, ...]) -> None:
        self._items: dict[str, EvidenceItem] = {}
        self._allowed_hosts = allowed_hosts

    def add(self, items: list[EvidenceItem]) -> list[EvidenceItem]:
        accepted = []
        for item in items:
            if item.quote.strip() and official_source(item.url, self._allowed_hosts):
                self._items[item.evidence_id] = item
                accepted.append(item)
        return accepted

    def resolve(self, evidence_ids: list[str]) -> list[EvidenceItem] | None:
        """Return the items, or None when any ID is unknown (invented citation)."""
        resolved = []
        for evidence_id in dict.fromkeys(evidence_ids):
            item = self._items.get(evidence_id)
            if item is None:
                return None
            resolved.append(item)
        return resolved

    def ids(self) -> list[str]:
        return list(self._items)

    def __len__(self) -> int:
        return len(self._items)


_SPLIT = re.compile(r"(?<=[.;])\s+|\n+")
_WORDS = re.compile(r"\w{4,}")


def extract_span(quote: str, question: str, limit: int) -> str:
    """Pick the sentences of an exact quote that best match the question (original order kept)."""
    text = quote.strip()
    if len(text) <= limit:
        return text
    wanted = set(_WORDS.findall(_normalize(question)))
    parts = [part.strip() for part in _SPLIT.split(text) if part.strip()]
    scored = sorted(
        range(len(parts)),
        key=lambda i: (-len(wanted & set(_WORDS.findall(_normalize(parts[i])))), i),
    )
    chosen: list[int] = []
    used = 0
    for index in scored:
        if used + len(parts[index]) + 1 > limit and chosen:
            continue
        chosen.append(index)
        used += len(parts[index]) + 1
        if used >= limit:
            break
    chosen.sort()
    picked = " ... ".join(parts[i][:limit] for i in chosen)
    return picked if len(chosen) == len(parts) else picked + " ..."


def render_answer(items: list[EvidenceItem], max_quote_chars: int, question: str = "") -> str:
    """Deterministic rendering: exact source quotes with inline markers. No model prose."""
    lines = []
    for item in items:
        quote = extract_span(item.quote, question, max_quote_chars)
        lines.append(f'- {item.medicine_name}, section {item.section}: "{quote}" [{item.evidence_id}]')
    return "\n".join(lines)
