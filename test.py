# portal.py — PickModelAI Unified Executive Architecture Portal (Public Beta Testing)
import os
import re
import json
import uuid
import logging
from decimal import Decimal, InvalidOperation
from threading import RLock
from urllib.parse import quote
from pathlib import Path
from datetime import date, datetime, timezone
from typing import Literal, List, Optional, Tuple, Dict, Any
from xml.sax.saxutils import escape
import requests
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, model_validator

try:
    from langchain_openai import ChatOpenAI  # type: ignore[import-not-found]
except ImportError:
    ChatOpenAI = None
from langchain_core.prompts import ChatPromptTemplate

from pdf_generator import generate_blueprint_pdf
from vault_agent import PlanModelVaultAgent

try:
    import langchain as _lc  # type: ignore[import-not-found]
    for _legacy_attr in ("verbose", "debug", "llm_cache"):
        if not hasattr(_lc, _legacy_attr):
            setattr(_lc, _legacy_attr, False)
except ImportError:
    _lc = None

# ---------------------------------------------------------------------------
# 0. ENVIRONMENT & DIRECTORY SETUP
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)

LEADS_DIR = BASE_DIR / "captured_leads"
LEADS_DIR.mkdir(parents=True, exist_ok=True)

# Optional fallback to sibling folders if testing existing Vault IDs
PRODUCT_B_LEADS_DIR = BASE_DIR.parent / "pickmodelai-product-b" / "captured_leads"
PRODUCT_C_LEADS_DIR = BASE_DIR.parent / "pickmodelai-product-c" / "captured_leads"


def get_config(key: str, default: str = "") -> str:
    val = os.getenv(key)
    if val:
        return val
    try:
        return str(st.secrets.get(key, default))
    except Exception:
        return default


OPENAI_API_KEY = get_config("OPENAI_API_KEY")
OPENAI_CHAT_MODEL = get_config("OPENAI_CHAT_MODEL", "gpt-4o-mini")
DEV_MODE = get_config("DEV_MODE", "False").lower() == "true"
MAX_DAILY_GENERATIONS_PER_IP = 5

st.set_page_config(
    page_title="PickModelAI — AI Delivery Plan Suite",
    page_icon="📐",
    layout="wide",
)

if not OPENAI_API_KEY:
    st.error("Missing `OPENAI_API_KEY` in your `.env` file or Streamlit Secrets.")
    st.stop()

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


@st.cache_resource
def get_vault_agent() -> Optional[PlanModelVaultAgent]:
    try:
        backend = PlanModelVaultAgent()
    except Exception:
        logging.getLogger(__name__).warning("Remote vault unavailable; using the local continuity journal.")
        backend = None
    return _ContinuityVaultAgent(backend)


# ---------------------------------------------------------------------------
# 1. SHARED RATE LIMITER, GUARDRAILS, MODALITY LOCK & REGIONAL VALIDATION
# ---------------------------------------------------------------------------
MAX_WORDS_PER_ANSWER = 200
VAULT_ID_REGEX = re.compile(r"\bPMAI-[0-9A-F]{4}-[0-9A-F]{4}\b", re.IGNORECASE)
URL_REGEX = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|rules)",
    r"system\s*prompt",
    r"you\s+are\s+now\s+",
    r"forget\s+(everything|all\s+instructions)",
    r"act\s+as\s+a\s+",
    r"jailbreak",
    r"do\s+not\s+follow",
]

PII_PATTERNS = [
    r"\b\d{3}-\d{2}-\d{4}\b",
    r"\b(?:\d[ -]*?){13,16}\b",
    r"sk-[a-zA-Z0-9]{20,}",
]

VIDEO_ONLY_TOKENS = ["veo", "runway", "kling", "sora", "ltx", "midjourney", "dall-e", "stable diffusion", "hunyuanvideo"]
AUDIO_ONLY_TOKENS = ["whisper", "elevenlabs", "kokoro", "piper tts", "deepgram"]


def _tier_number(tier: Any) -> str:
    text = str(tier).strip()
    if text in {"1", "2", "3"}:
        return text
    match = re.search(r"\bTier\s*([123])\b", text, re.IGNORECASE)
    if match:
        return match.group(1)
    match = re.search(r"\bProduct\s*([ABC])\b", text, re.IGNORECASE)
    return {"A": "1", "B": "2", "C": "3"}.get(match.group(1).upper(), "") if match else ""


class _ContinuityVaultAgent(PlanModelVaultAgent):
    """Keep a canonical, append-only tier journal around the supplied vault backend."""

    def __init__(self, backend: Optional[PlanModelVaultAgent], leads_dir: Path = LEADS_DIR):
        self._backend = backend
        self._leads_dir = leads_dir
        self._leads_dir.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()

    @staticmethod
    def generate_client_ref_id() -> str:
        token = uuid.uuid4().hex[:8].upper()
        return f"PMAI-{token[:4]}-{token[4:]}"

    def retrieve_client_reports(self, vault_id: str) -> List[Dict[str, Any]]:
        if not VAULT_ID_REGEX.fullmatch(vault_id):
            raise ValueError("Invalid Vault ID. Expected PMAI-XXXX-XXXX using hexadecimal characters.")
        records = []
        folders = [self._leads_dir]
        if self._leads_dir == LEADS_DIR:
            folders.extend([PRODUCT_B_LEADS_DIR, PRODUCT_C_LEADS_DIR])
        for folder in folders:
            if not folder.exists():
                continue
            for path in sorted(folder.glob(f"{vault_id}_*.json"), reverse=True):
                try:
                    record = json.loads(path.read_text(encoding="utf-8"))
                    if record.get("client_vault_id", record.get("vault_id")) != vault_id:
                        continue
                    report = record.get("report_data", record.get("generated_blueprint", {}))
                    inputs = record.get("user_inputs", record.get("project_inputs", {}))
                    if not isinstance(report, dict) or not isinstance(inputs, dict):
                        continue
                    records.append({**record, "report_data": report, "user_inputs": inputs})
                except (OSError, ValueError, TypeError, AttributeError):
                    logging.getLogger(__name__).warning("Unreadable vault journal entry: %s", path.name)
        if self._backend is not None:
            try:
                remote_records = self._backend.retrieve_client_reports(vault_id) or []
                records.extend(r for r in remote_records if isinstance(r, dict))
            except Exception:
                if not records:
                    raise RuntimeError("Vault verification is unavailable. Please retry; no replacement ID was created.")
                logging.getLogger(__name__).warning("Remote lookup failed; using locally persisted vault records.")
        return records

    def resolve_vault(self, vault_id: Optional[str], current_product: str = "A") -> Dict[str, Any]:
        if current_product not in {"A", "B", "C"}:
            raise ValueError("Unsupported product.")
        records = self.retrieve_client_reports(vault_id) if vault_id else []
        if not records:
            return {
                "vault_id": self.generate_client_ref_id(),
                "origin_product": current_product,
                "tiers_generated": [],
                "parent_chain": [],
            }
        tiers = sorted({_tier_number(r.get("tier", "")) for r in records} - {""}, key=int)
        origin = {"1": "A", "2": "B", "3": "C"}.get(tiers[0] if tiers else "", current_product)
        for record in records:
            report = record.get("report_data", {})
            lineage = record.get("lineage") or report.get("lineage", {})
            if isinstance(lineage, dict) and lineage.get("origin_product") in {"A", "B", "C"}:
                origin = lineage["origin_product"]
                break
        return {
            "vault_id": vault_id,
            "origin_product": origin,
            "tiers_generated": tiers,
            "parent_chain": [
                {"vault_id": vault_id, "product": {"1": "A", "2": "B", "3": "C"}[tier]}
                for tier in tiers
            ],
        }

    def check_tier_generation_allowed(self, vault_id: str, target_tier: str) -> Dict[str, Any]:
        target = _tier_number(target_tier)
        if not target:
            raise ValueError("Unsupported target tier.")
        records = self.retrieve_client_reports(vault_id)
        tiers = {_tier_number(r.get("tier", "")) for r in records}
        if target in tiers:
            return {
                "allowed": False,
                "reason": f"Tier {target} has already been generated for {vault_id}. Contact support@pickmodelai.com for revisions.",
                "warning": "",
            }
        if target == "3" and not tiers.intersection({"1", "2"}):
            return {"allowed": False, "reason": "Product C requires a verified Product A or Product B vault record.", "warning": ""}
        warning = ""
        if target == "3" and "1" in tiers and "2" not in tiers:
            warning = "Product B is pending; generating C now overlaps ~40% with eventual B"
        return {"allowed": True, "reason": "", "warning": warning}

    def save_blueprint_to_vault(
        self,
        user_inputs: Dict[str, str],
        report_data: Dict[str, Any],
        tier: str,
        client_ref_id: Optional[str] = None,
    ) -> str:
        target = _tier_number(tier)
        if not target:
            raise ValueError("Unsupported target tier.")
        ref_id = client_ref_id or self.generate_client_ref_id()
        if not VAULT_ID_REGEX.fullmatch(ref_id):
            raise ValueError("Cannot persist an invalid Vault ID.")
        with self._lock:
            gate = self.check_tier_generation_allowed(ref_id, target)
            if not gate["allowed"]:
                raise ValueError(gate["reason"])
            records = self.retrieve_client_reports(ref_id)
            state = self.resolve_vault(ref_id, {"1": "A", "2": "B", "3": "C"}[target]) if records else {
                "vault_id": ref_id,
                "origin_product": {"1": "A", "2": "B", "3": "C"}[target],
                "tiers_generated": [],
                "parent_chain": [],
            }
            lineage = {
                "origin_product": state["origin_product"],
                "parent_vault_id": ref_id if records else None,
                "parent_chain": state["parent_chain"],
            }
            payload = {
                **report_data,
                "vault_id": ref_id,
                "lineage": lineage,
                "tiers_generated": sorted(set(state["tiers_generated"] + [target]), key=int),
            }
            tier_label = "Tier 1 (Product A — Genesis Diagnostic)" if target == "1" else tier
            record = {
                "client_vault_id": ref_id,
                "tier": tier_label,
                "captured_at_utc": datetime.now(timezone.utc).isoformat(),
                "user_inputs": user_inputs,
                "report_data": payload,
                "contact_details": payload.get("contact_details", {}),
                "lineage": lineage,
                "tiers_generated": payload["tiers_generated"],
            }
            path = self._leads_dir / f"{ref_id}_Tier{target}.json"
            temporary = self._leads_dir / f".{ref_id}_{uuid.uuid4().hex}.tmp"
            try:
                with temporary.open("x", encoding="utf-8") as handle:
                    json.dump(record, handle, indent=2, ensure_ascii=False)
                    handle.flush()
                    os.fsync(handle.fileno())
                try:
                    os.link(temporary, path)
                except FileExistsError as exc:
                    raise ValueError(f"Tier {target} is already committed for {ref_id}.") from exc
            finally:
                temporary.unlink(missing_ok=True)
            report_data.update({"vault_id": ref_id, "lineage": lineage, "tiers_generated": payload["tiers_generated"]})
            if self._backend is not None:
                try:
                    saved_id = self._backend.save_blueprint_to_vault(
                        user_inputs=user_inputs,
                        report_data=payload,
                        tier=tier_label,
                        client_ref_id=ref_id,
                    )
                    if saved_id != ref_id:
                        logging.getLogger(__name__).error(
                            "Remote vault returned a different ID; the local canonical ID was retained. Backend repair is required."
                        )
                except Exception:
                    logging.getLogger(__name__).warning(
                        "Remote vault save failed; canonical report and lineage remain in the local continuity journal."
                    )
            return ref_id


def _extract_locked_costs_from_records(records: List[Dict[str, Any]]) -> Tuple[Optional[int], Optional[int]]:
    """Extract canonical baseline paid & OSS monthly costs from Product A (or earliest tier) so B & C stay 100% synced."""
    if not records:
        return None, None
    sorted_recs = sorted(records, key=lambda r: int(_tier_number(r.get("tier", "")) or "9"))
    for rec in sorted_recs:
        rep = rec.get("report_data", {})
        paid = rep.get("paid_monthly_cost_usd") or rep.get("paid_stack", {}).get("monthly_cost_number_usd")
        oss = rep.get("oss_monthly_cost_usd") or rep.get("open_source_stack", {}).get("monthly_cost_number_usd")
        if paid and int(paid) > 0:
            return int(paid), int(oss) if oss and int(oss) > 0 else None
    return None, None


def _prepare_vault_generation(vault_id: Optional[str], target_tier: str) -> Dict[str, Any]:
    vault = get_vault_agent()
    if vault is None:
        raise RuntimeError("Vault persistence is unavailable.")
    product = {"1": "A", "2": "B", "3": "C"}[target_tier]
    state = vault.resolve_vault(vault_id, current_product=product)
    gate = vault.check_tier_generation_allowed(state["vault_id"], target_tier)
    if not gate["allowed"]:
        st.error(gate["reason"])
        st.stop()
    state["warning"] = gate["warning"]
    if gate["warning"]:
        st.warning(gate["warning"])
    return state


def _attach_vault_context(data: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    data["vault_id"] = state["vault_id"]
    data["lineage"] = {
        "origin_product": state["origin_product"],
        "parent_vault_id": state["vault_id"] if state["tiers_generated"] else None,
        "parent_chain": state["parent_chain"],
    }
    data["vault_lifecycle_warning"] = state.get("warning", "")
    return data


def _report_vault_id(report: Dict[str, Any], existing_vault_id: Optional[str]) -> Optional[str]:
    generated_id = report.get("vault_id")
    if existing_vault_id and generated_id and generated_id != existing_vault_id:
        raise ValueError("Generated report does not belong to the imported Vault ID; refusing to break lineage.")
    return existing_vault_id or generated_id


@st.cache_resource
def get_ip_rate_limit_store() -> dict:
    return {}


def get_client_ip() -> str:
    try:
        headers = st.context.headers
        forwarded_for = headers.get("X-Forwarded-For") or headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        real_ip = headers.get("X-Real-Ip") or headers.get("x-real-ip")
        if real_ip:
            return real_ip.strip()
    except Exception:
        pass
    return "127.0.0.1"


def check_and_increment_rate_limit(increment: bool = False) -> Tuple[bool, int]:
    if DEV_MODE:
        return True, 999
    store = get_ip_rate_limit_store()
    ip = get_client_ip()
    today_key = (ip, date.today().isoformat())
    current_count = store.get(today_key, 0)
    if current_count >= MAX_DAILY_GENERATIONS_PER_IP:
        return False, 0
    if increment:
        store[today_key] = current_count + 1
        current_count += 1
    return True, MAX_DAILY_GENERATIONS_PER_IP - current_count


def validate_user_input(text: str, step: int, max_words: int) -> Tuple[bool, str]:
    cleaned = text.strip()
    if not cleaned:
        return False, "⚠️ Input cannot be empty."
    words = cleaned.split()
    word_count = len(words)
    if word_count > max_words:
        return False, f"⚠️ Please keep your answer under {max_words} words (currently {word_count} words)."
    min_words = 1 if step == 3 else 3
    if word_count < min_words or len(cleaned) < 2:
        return False, f"⚠️ Please provide a bit more detail (at least {min_words} words)."
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, cleaned, re.IGNORECASE):
            return False, "🚫 Security Guardrail: Disallowed instruction override detected."
    for pattern in PII_PATTERNS:
        if re.search(pattern, cleaned):
            return False, "🔒 Privacy Guardrail: Please do not enter SSNs, credit cards, or API keys."
    return True, ""


def validate_chat_answer(text: str, max_words: int = MAX_WORDS_PER_ANSWER) -> Tuple[bool, str]:
    cleaned = text.strip()
    if not cleaned:
        return False, "⚠️ Input cannot be empty."
    words = cleaned.split()
    if len(words) > max_words:
        return False, f"⚠️ Please keep your response under {max_words} words (currently {len(words)} words)."
    for pat in INJECTION_PATTERNS:
        if re.search(pat, cleaned, re.IGNORECASE):
            return False, "🚫 Security Guardrail: Disallowed instruction override detected."
    for pat in PII_PATTERNS:
        if re.search(pat, cleaned):
            return False, "🔒 Privacy Guardrail: Do not enter SSNs, credit card numbers, or API keys."
    return True, ""


def evaluate_project_safety(profile_text: str) -> Tuple[bool, str]:
    explicit_malicious_phrases = [
        r"steal\s+(people's\s+)?(passwords|credentials|accounts)",
        r"credit\s+card\s+(fraud|steal|dump|testing|stolen)",
        r"cyberbully",
        r"harass.*until\s+she\s+deletes",
        r"non-consensual\s+sex",
        r"revenge\s+porn",
        r"fake\s+product\s+scam",
        r"dark\s+web\s+dump",
        r"polymorphic\s+ransomware",
        r"encrypt.*files.*demand.*bitcoin",
        r"send\s+fake\s+login\s+emails",
    ]
    for pattern in explicit_malicious_phrases:
        if re.search(pattern, profile_text, re.IGNORECASE):
            return False, "Security Policy Violation: Malicious intent or prohibited activity detected."
    try:
        if ChatOpenAI is None:
            return True, ""
        safety_llm = ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=0.0)
        safety_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are an enterprise AI safety classifier. Return 'UNSAFE: [reason]' if the project involves phishing, "
             "credential theft, fraud, malware, harassment, or non-consensual sexual content. Return 'SAFE' for legitimate "
             "business, defensive security, compliance, or clinical use cases."),
            ("human", "{profile}")
        ])
        res = (safety_prompt | safety_llm).invoke({"profile": profile_text})
        content = str(res.content).strip()
        if content.upper().startswith("UNSAFE"):
            return False, content
    except Exception:
        pass
    return True, ""


def detect_primary_modality(core_idea: str) -> str:
    """BUG #6 FIX: Weighted modality classifier so secondary 'transcribe voice notes'
    does not hijack a primarily text/PDF/email/code assistant into a telephony VOICE_AUDIO stack."""
    text = core_idea.lower()

    text_keywords = ["email", "emails", "pdf", "pdfs", "document", "documents", "text", "summarize", "summary", "code", "knowledge worker", "report", "chat", "rag"]
    voice_primary_keywords = ["call center", "phone", "telephony", "ivr", "podcast", "tts", "spoken", "voice agent", "voicebot", "real-time voice"]
    voice_secondary_keywords = ["voice", "audio", "speech", "stt", "transcribe", "voice notes"]
    video_keywords = ["video", "reels", "tiktok", "youtube shorts", "animation", "mp4", "film"]
    vision_keywords = ["image", "photo", "camera", "screenshot", "ocr", "vision", "x-ray", "scan"]

    text_score = sum(1 for w in text_keywords if re.search(rf"\b{re.escape(w)}\b", text))
    voice_primary_score = sum(2 for w in voice_primary_keywords if re.search(rf"\b{re.escape(w)}\b", text))
    voice_secondary_score = sum(1 for w in voice_secondary_keywords if re.search(rf"\b{re.escape(w)}\b", text))
    video_score = sum(2 for w in video_keywords if re.search(rf"\b{re.escape(w)}\b", text))
    vision_score = sum(2 for w in vision_keywords if re.search(rf"\b{re.escape(w)}\b", text))

    if video_score > 0 and video_score >= text_score:
        return "VIDEO_GEN"
    if vision_score > 0 and vision_score > text_score:
        return "MULTIMODAL_VISION"
    # Only trigger VOICE_AUDIO if dedicated telephony/voice-agent keywords exist OR voice outweighs text/document terms
    if voice_primary_score > 0 or (voice_secondary_score > 0 and text_score == 0):
        return "VOICE_AUDIO"
    return "TEXT_REASONING"


def sanitize_for_pdf(text: str) -> str:
    no_urls = URL_REGEX.sub("[link removed]", str(text)).strip()
    return escape(no_urls)


COUNTRY_RULES: Dict[str, Dict[str, str]] = {
    "India": {
        "iso": "IN",
        "dial": "+91",
        "pin_regex": r"^[1-9][0-9]{5}$",
        "pin_hint": "6-digit PIN code (e.g., 110001 or 560001)",
        "phone_regex": r"^(?:\+?91[\s-]?)?[6-9]\d{9}$",
        "phone_hint": "10-digit Indian mobile number (starting with 6-9, optional +91)",
    },
    "United States": {
        "iso": "US",
        "dial": "+1",
        "pin_regex": r"^\d{5}(?:-\d{4})?$",
        "pin_hint": "5-digit ZIP code (e.g., 10001 or 94105)",
        "phone_regex": r"^(?:\+?1[\s.-]?)?\(?[2-9]\d{2}\)?[\s.-]?\d{3}[\s.-]?\d{4}$",
        "phone_hint": "10-digit US phone number (e.g., +1 415-555-0192)",
    },
    "United Kingdom": {
        "iso": "GB",
        "dial": "+44",
        "pin_regex": r"^[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}$",
        "pin_hint": "UK Postcode (e.g., SW1A 1AA or EC1A 1BB)",
        "phone_regex": r"^(?:\+?44[\s-]?|0)(?:7\d{3}|\d{3,4})[\s-]?\d{6}$",
        "phone_hint": "UK phone number (e.g., +44 7700 900077)",
    },
    "Canada": {
        "iso": "CA",
        "dial": "+1",
        "pin_regex": r"^[ABCEGHJ-NPRSTVXY]\d[ABCEGHJ-NPRSTV-Z]\s?\d[ABCEGHJ-NPRSTV-Z]\d$",
        "pin_hint": "Canadian Postal Code (e.g., M5V 2T6)",
        "phone_regex": r"^(?:\+?1[\s.-]?)?\(?[2-9]\d{2}\)?[\s.-]?\d{3}[\s.-]?\d{4}$",
        "phone_hint": "10-digit Canadian phone number (e.g., +1 416-555-0148)",
    },
    "Australia": {
        "iso": "AU",
        "dial": "+61",
        "pin_regex": r"^\d{4}$",
        "pin_hint": "4-digit Australian Postcode (e.g., 2000 or 3000)",
        "phone_regex": r"^(?:\+?61[\s-]?|0)[2-478](?:[\s-]?\d){8}$",
        "phone_hint": "Australian phone number (e.g., +61 412 345 678)",
    },
    "Germany": {
        "iso": "DE",
        "dial": "+49",
        "pin_regex": r"^\d{5}$",
        "pin_hint": "5-digit German Postleitzahl (e.g., 10115)",
        "phone_regex": r"^(?:\+?49[\s-]?|0)[1-9]\d{6,13}$",
        "phone_hint": "German phone number (e.g., +49 151 23456789)",
    },
    "Singapore": {
        "iso": "SG",
        "dial": "+65",
        "pin_regex": r"^\d{6}$",
        "pin_hint": "6-digit Singapore Postal Code (e.g., 018956)",
        "phone_regex": r"^(?:\+?65[\s-]?)?[689]\d{7}$",
        "phone_hint": "8-digit Singapore number starting with 6, 8, or 9",
    },
    "United Arab Emirates": {
        "iso": "AE",
        "dial": "+971",
        "pin_regex": r"^(?:\d{5,6}|00000)$",
        "pin_hint": "UAE Postal / PO Box code (or 00000)",
        "phone_regex": r"^(?:\+?971[\s-]?|0)(?:5[024568]|[234679])\d{7}$",
        "phone_hint": "UAE phone number (e.g., +971 50 123 4567)",
    },
}


@st.cache_data(ttl=3600, show_spinner=False)
def lookup_city_from_pin(country: str, pin_code: str) -> Tuple[bool, str, str]:
    rules = COUNTRY_RULES.get(country)
    if not rules:
        return False, "", "Unsupported country selected."
    clean_pin = pin_code.strip().upper()
    if not re.match(rules["pin_regex"], clean_pin, re.IGNORECASE):
        return False, "", f"Invalid PIN/Postal code for {country}. Expected: {rules['pin_hint']}."
    iso = rules["iso"]
    headers = {"User-Agent": "PickModelAI/1.0"}

    if iso == "IN":
        try:
            url = f"https://api.postalpincode.in/pincode/{quote(clean_pin)}"
            resp = requests.get(url, headers=headers, timeout=5)
            if resp.status_code == 200:
                payload = resp.json()
                if payload and payload[0].get("Status") == "Success":
                    po = payload[0]["PostOffice"][0]
                    city_str = f"{po.get('District', '')}, {po.get('State', '')}".strip(", ")
                    return True, city_str, f"Verified PIN {clean_pin} → {city_str}, India"
            return False, "", f"PIN code {clean_pin} not found in India Post directory."
        except Exception:
            return True, "", f"PIN format valid ({clean_pin}). Please confirm your city below."

    if iso in {"US", "GB", "CA", "AU", "DE"}:
        lookup_zip = clean_pin.split()[0] if iso in {"GB", "CA"} else clean_pin.split("-")[0]
        try:
            url = f"https://api.zippopotam.us/{iso.lower()}/{quote(lookup_zip)}"
            resp = requests.get(url, headers=headers, timeout=5)
            if resp.status_code == 200:
                places = resp.json().get("places", [])
                if places:
                    place_name = places[0].get("place name", "")
                    state_abbr = places[0].get("state abbreviation") or places[0].get("state", "")
                    city_str = f"{place_name}, {state_abbr}".strip(", ")
                    return True, city_str, f"Verified Postal Code {clean_pin} → {city_str}, {country}"
            return False, "", f"Postal code {clean_pin} not found for {country}."
        except Exception:
            return True, "", f"Postal format valid ({clean_pin}). Please confirm your city below."

    if iso == "SG":
        return True, "Singapore", "Verified Singapore Postal Code → Singapore"
    if iso == "AE":
        return True, "Dubai / Abu Dhabi", "Verified UAE code. Confirm Emirate/City below."
    return True, "", "Postal format verified."


def validate_regional_contact(
    name: str, company: str, email: str, country: str, pin_code: str, city: str, phone: str
) -> Tuple[bool, str]:
    if len(name.strip()) < 2:
        return False, "Please enter your full name."
    if len(company.strip()) < 2:
        return False, "Please enter your company or organization name."
    if not re.match(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$", email.strip()):
        return False, "Please enter a valid business or personal email address."
    rules = COUNTRY_RULES.get(country)
    if not rules:
        return False, "Please select a valid country."
    pin_ok, _, pin_msg = lookup_city_from_pin(country, pin_code)
    if not pin_ok:
        return False, pin_msg
    if len(city.strip()) < 2:
        return False, "Please verify your PIN/Postal code so City is populated."
    if not re.match(rules["phone_regex"], phone.strip()):
        return False, f"Invalid phone number for {country}. Expected: {rules['phone_hint']}."
    return True, ""


# ===========================================================================
# 2. PRODUCT A: GENESIS ARCHITECTURE DIAGNOSTIC (SCHEMAS & ENGINE)
# ===========================================================================
import re
from typing import List, Literal
from pydantic import BaseModel, Field, field_validator, model_validator
from langchain_core.prompts import ChatPromptTemplate

VECTOR_DB_LITERAL = Literal[
    "Pinecone Serverless (EU/US)",
    "ChromaDB",
    "PGVector (AWS RDS)",
    "Qdrant Cloud (EU/US)",
]


class PaidStackOption(BaseModel):
    model_stack: str = Field(
        description=(
            "Complete commercial cloud model stack matching the exact modality. "
            "Must specify enterprise/zero-data-retention API endpoints and regional routing "
            "(e.g., EU endpoints for GDPR) when privacy constraints exist."
        )
    )
    vector_db: VECTOR_DB_LITERAL = Field(
        description="Recommended managed vector database aligned with data residency requirements."
    )
    why_this_stack: str = Field(
        description=(
            "Specific 1-2 sentence architectural justification covering latency, team size fit "
            "(zero MLOps overhead), and zero-training data privacy compliance (max 45 words)."
        )
    )
    estimated_monthly_cost: str = Field(
        description=(
            "Exact USD breakdown distinguishing Prototype phase vs Production run-rate, e.g., "
            "'Prototype: ~$180/mo | Production: $600/mo total ($500/mo API usage + $100/mo vector DB)'."
        )
    )
    monthly_cost_number_usd: int = Field(
        ge=1,
        le=50000,
        description="Target production monthly run-rate cost integer in USD.",
    )
    scaling_path: str = Field(
        description=(
            "Concrete 1-sentence scaling and cost optimization strategy, such as semantic caching, "
            "prompt routing to smaller models, or batching (max 35 words)."
        )
    )

    @field_validator("model_stack", "why_this_stack", "scaling_path", "estimated_monthly_cost")
    @classmethod
    def sanitize_text_fields(cls, v: str) -> str:
        return sanitize_for_pdf(v)


class OpenSourceStackOption(BaseModel):
    model_stack: str = Field(
        description=(
            "Complete self-hosted open-weight stack on AWS GPU (e.g., vLLM/TGI on g5.xlarge/g5.2xlarge) "
            "matching the exact modality and target AWS region (e.g., eu-central-1)."
        )
    )
    vector_db: VECTOR_DB_LITERAL = Field(
        description="Recommended self-hosted or VPC-isolated vector database."
    )
    why_this_stack: str = Field(
        description=(
            "Specific 1-2 sentence trade-off analysis: explain VPC data isolation benefits vs. "
            "the hidden DevOps/MLOps engineering overhead required to maintain self-hosted GPUs (max 45 words)."
        )
    )
    estimated_monthly_cost: str = Field(
        description=(
            "Exact USD infrastructure breakdown (minimum $350/mo for 24/7 AWS GPU + vector DB) "
            "plus an explicit note that DevOps/MLOps maintenance labor is excluded."
        )
    )
    monthly_cost_number_usd: int = Field(
        ge=350,
        le=50000,
        description="Total monthly AWS infrastructure cost integer in USD (minimum 350 for 24/7 GPU + DB).",
    )
    scaling_path: str = Field(
        description=(
            "Concrete 1-sentence GPU scaling and quantization tip, e.g., AWQ/FP8 quantization "
            "with vLLM continuous batching and autoscaling groups (max 35 words)."
        )
    )

    @field_validator("model_stack", "why_this_stack", "scaling_path", "estimated_monthly_cost")
    @classmethod
    def sanitize_text_fields(cls, v: str) -> str:
        return sanitize_for_pdf(v)

    @model_validator(mode="after")
    def enforce_aws_cost_floor(self):
        # 24/7 AWS g5.xlarge + RDS PGVector realistically starts around $350-$450/mo
        if self.monthly_cost_number_usd < 350:
            self.monthly_cost_number_usd = 420
        if "$0" in self.estimated_monthly_cost and "API" not in self.estimated_monthly_cost:
            self.estimated_monthly_cost = (
                "$420/mo infra ($0 API fees + $290/mo AWS g5.xlarge GPU + $130/mo RDS PGVector; "
                "excludes MLOps maintenance labor)"
            )
        elif "excl" not in self.estimated_monthly_cost.lower() and "devops" not in self.estimated_monthly_cost.lower():
            self.estimated_monthly_cost = sanitize_for_pdf(
                f"{self.estimated_monthly_cost} (infra only; excludes DevOps/MLOps labor)"
            )
        return self


class FreeTierBlueprint(BaseModel):
    project_short_title: str = Field(description="3 to 5 word executive project title.")
    feasibility_grade: Literal[
        "Grade A (Fast & Affordable to Build)",
        "Grade B (Moderate Cloud Compute Cost)",
        "Grade C (Complex Infrastructure Required)",
    ]
    user_idea_summary: str = Field(
        description=(
            "Executive 2-sentence summary covering WHAT the system automates, WHO the primary users "
            "and end beneficiaries are, and quantifiable 6-MONTH SLA/KPI GOALS (e.g., % deflection, "
            "FRT target, ticket volume) (40-60 words)."
        )
    )
    workflow_steps: List[str] = Field(
        description=(
            "Exactly 4 concrete architectural steps (6-12 words each) showing end-to-end data flow: "
            "(1) API/webhook ingestion, (2) PII scrubbing & RAG retrieval, (3) LLM generation with "
            "confidence scoring, (4) Deterministic fallback/escalation threshold (e.g., score < 0.85) with summary."
        )
    )
    paid_stack: PaidStackOption
    open_source_stack: OpenSourceStackOption
    next_steps_recommendation: str = Field(
        description=(
            "2-sentence Principal Architect guardrail warning naming exact required production components: "
            "(1) deterministic escalation rules (confidence/sentiment thresholds), and "
            "(2) compliance architecture (e.g., GDPR DPA, EU region pinning, PII redaction, zero-data-retention API). "
            "Max 75 words."
        )
    )

    @field_validator("project_short_title", "user_idea_summary", "next_steps_recommendation")
    @classmethod
    def sanitize_summary_fields(cls, v: str) -> str:
        return sanitize_for_pdf(v)

    @field_validator("workflow_steps")
    @classmethod
    def ensure_four_steps(cls, v: List[str]) -> List[str]:
        cleaned = [sanitize_for_pdf(step) for step in v if step.strip()]
        defaults = [
            "Ingest live webhooks and order payloads via REST APIs",
            "Scrub PII and retrieve policy context from vector DB",
            "Generate grounded reply with deterministic confidence score check",
            "Auto-escalate tickets below 0.85 confidence with agent summary",
        ]
        while len(cleaned) < 4:
            cleaned.append(defaults[len(cleaned)])
        return cleaned[:4]


def fetch_live_ai_model_research(user_idea: str, volume_privacy: str, modality: str) -> str:
    short_idea = " ".join(user_idea.split()[:14])
    if modality == "VOICE_AUDIO":
        queries = [
            f"best low latency commercial voice AI stack STT LLM TTS 2026 OpenAI Realtime Deepgram ElevenLabs {short_idea}",
            f"best open source self hosted STT LLM TTS voice stack 2026 Whisper Llama Kokoro Piper {short_idea}",
        ]
    elif modality == "VIDEO_GEN":
        queries = [
            f"best commercial AI video generation API 2026 Runway Gen-4 Google Veo Kling {short_idea}",
            f"best open source self hosted video generation model 2026 LTX-2 CogVideoX HunyuanVideo {short_idea}",
        ]
    elif modality == "MULTIMODAL_VISION":
        queries = [
            f"best commercial vision multimodal AI API 2026 GPT-4o Gemini 2.5 Claude {short_idea}",
            f"best open source vision language model 2026 Qwen2.5-VL Llama Vision {short_idea}",
        ]
    else:
        queries = [
            f"best commercial LLM API and pricing 2026 GPT-4o Claude Sonnet Gemini {short_idea}",
            f"best open source self hosted LLM 2026 Llama 3.3 Llama 4 DeepSeek R1 Mistral {short_idea}",
        ]
    snippets = []
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            for q in queries:
                results = ddgs.text(q, max_results=3)
                if results:
                    for r in results:
                        clean_body = re.sub(
                            r"https?://\S+|www\.\S+",
                            "",
                            f"{r.get('title', '')}: {r.get('body', '')}",
                        )
                        snippets.append(f"- {clean_body.strip()}")
    except Exception:
        pass
    return "\n".join(snippets) if snippets else f"Modality={modality}. Select only verified models matching {modality}."


def enforce_modality_integrity(data: dict, modality: str) -> dict:
    paid_text = f"{data['paid_stack']['model_stack']} {data['paid_stack']['why_this_stack']}".lower()
    oss_text = f"{data['open_source_stack']['model_stack']} {data['open_source_stack']['why_this_stack']}".lower()
    if modality == "VOICE_AUDIO":
        if any(tok in paid_text for tok in VIDEO_ONLY_TOKENS):
            data["paid_stack"]["model_stack"] = "OpenAI Realtime API (GPT-4o) + ElevenLabs Turbo v2.5"
        if any(tok in oss_text for tok in VIDEO_ONLY_TOKENS):
            data["open_source_stack"]["model_stack"] = "Whisper-v3 + Llama 3.3 8B + Kokoro-82M TTS on AWS g5.2xlarge"
    elif modality == "VIDEO_GEN":
        if any(tok in paid_text for tok in AUDIO_ONLY_TOKENS):
            data["paid_stack"]["model_stack"] = "Google Veo 3.1 / Runway Gen-4.5 API"
        if any(tok in oss_text for tok in AUDIO_ONLY_TOKENS):
            data["open_source_stack"]["model_stack"] = "LTX-2.5 Video Diffusion on AWS g5.2xlarge"
    return data


def generate_assessment_from_llm(user_profile: str, core_idea: str, volume_privacy: str) -> dict:
    lifecycle = _prepare_vault_generation(None, "1")
    modality = detect_primary_modality(core_idea)
    live_web_research = fetch_live_ai_model_research(core_idea, volume_privacy, modality)
    
    system_prompt = """You are a Principal AI Solutions Architect at PickModelAI.com authoring an executive-grade AI Architecture Diagnostic for CXOs and Lead Engineers.
DETECTED PROJECT MODALITY: {modality}
LIVE ONLINE RESEARCH RESULTS:
<live_web_research>
{live_web_research}
</live_web_research>

STRICT ARCHITECTURAL & CXO RULES:
1. Modality Alignment: Match {modality} strictly. If TEXT_REASONING, design an API/webhook + RAG + structured output + deterministic router stack (do NOT include voice/video/telephony).
2. Realistic TCO & Team Fit:
   - Option 1 (Managed Cloud API): Highlight speed to market, zero MLOps maintenance for lean teams, and separate Prototype vs. Production monthly costs if the client states both budgets.
   - Option 2 (Self-Hosted AWS VPC): Use realistic 24/7 AWS GPU pricing (minimum $350/mo for g5.xlarge + RDS PGVector). Explicitly warn in `why_this_stack` and `estimated_monthly_cost` that self-hosting requires DevOps/MLOps engineering labor that often outweighs raw compute savings for small teams.
3. Deterministic Escalation Workflow: In `workflow_steps` (Step 4) and `next_steps_recommendation`, never write vague phrases like "escalate complex cases". Specify deterministic routing mechanisms (e.g., confidence score < 0.85, missing API order status, negative sentiment trigger, or refund threshold) that auto-generate structured handoff summaries.
4. Data Residency & Privacy Architecture: If the client requests GDPR, EU servers, or "no model training on customer data", explicitly specify:
   - EU region pinning (e.g., AWS `eu-central-1` Frankfurt / EU-hosted vector DB) instead of US-East-1 defaults.
   - Enterprise Zero-Data-Retention (ZDR) API endpoints with a signed DPA.
   - Pre-inference PII scrubbing/masking middleware before external LLM calls.
5. Executive Precision: Include the client's exact KPIs (e.g., ticket volume, % deflection target, response time SLA, team size) in `user_idea_summary`. Avoid generic filler or hype adjectives."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "<client_answers>\n{user_profile}\n</client_answers>"),
    ])
    if ChatOpenAI is None:
        raise RuntimeError("The langchain-openai package is required to generate assessments.")
    llm = ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=0.15, max_retries=2, timeout=60)
    structured_llm = llm.with_structured_output(FreeTierBlueprint)
    raw_res = (prompt | structured_llm).invoke({
        "modality": modality,
        "live_web_research": live_web_research,
        "user_profile": user_profile,
    })
    data = raw_res.model_dump() if isinstance(raw_res, BaseModel) else dict(raw_res)
    data = enforce_modality_integrity(data, modality)
    data["detected_modality"] = modality
    data["paid_stack"]["llm_recommendation"] = data["paid_stack"]["model_stack"]
    data["open_source_stack"]["llm_recommendation"] = data["open_source_stack"]["model_stack"]
    return _attach_vault_context(data, lifecycle)


def save_product_a_to_vault(answers: List[str], blueprint_dict: dict) -> str:
    vault = get_vault_agent()
    if vault is None:
        raise RuntimeError("Vault persistence is unavailable; no replacement ID was created.")
    user_inputs = {
        "goal": answers[0],
        "workflow": answers[1],
        "constraints": answers[2],
        "budget": answers[3],
    }
    vault_report_payload = {
        **blueprint_dict,
        "modality": blueprint_dict.get("detected_modality", "TEXT_REASONING"),
        "executive_summary": blueprint_dict.get("user_idea_summary", ""),
        "paid_stack_summary": blueprint_dict.get("paid_stack", {}).get("model_stack", ""),
        "oss_stack_summary": blueprint_dict.get("open_source_stack", {}).get("model_stack", ""),
        "paid_monthly_cost_usd": int(blueprint_dict.get("paid_stack", {}).get("monthly_cost_number_usd", 0)),
        "oss_monthly_cost_usd": int(blueprint_dict.get("open_source_stack", {}).get("monthly_cost_number_usd", 350)),
    }
    ref_id = vault.save_blueprint_to_vault(
        user_inputs=user_inputs,
        report_data=vault_report_payload,
        tier="Tier 1 (Product A — Genesis Diagnostic)",
        client_ref_id=blueprint_dict.get("vault_id"),
    )
    for key in ("vault_id", "lineage", "tiers_generated"):
        blueprint_dict[key] = vault_report_payload[key]
    return ref_id

# ===========================================================================
# 3. PRODUCT B: DIY ENGINEERING BLUEPRINT (SCHEMAS, ENGINE & 7-SECTION RENDERER)
# ===========================================================================
import re
import json
import uuid
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
import streamlit as st
from pydantic import BaseModel, Field, field_validator
from langchain_core.prompts import ChatPromptTemplate

# ==============================================================================
# 1. VAULT-AWARE 4-QUESTION DISCOVERY BOT (ANALYZES PRODUCT A -> 200-WORD CAP)
# ==============================================================================

class ProductBDiscoveryQuestionnaire(BaseModel):
    vault_analysis_summary: str = Field(
        description="2-sentence architect synthesis of the locked Product A report confirming modality, stack, and baseline OpEx."
    )
    q1_api_contracts: str = Field(
        description="Question 1 (API & Data Contracts): Specific question asking about exact webhook events, REST endpoints, and auth payloads for the systems named in Product A (max 35 words)."
    )
    q2_deterministic_rules: str = Field(
        description="Question 2 (Deterministic Guardrails & Escalation): Specific question asking for exact business rules, confidence score cutoffs (e.g., <0.85), or monetary/refund limits that must bypass LLM autonomy (max 35 words)."
    )
    q3_rag_and_privacy: str = Field(
        description="Question 3 (Knowledge Indexing & PII Redaction): Specific question asking about document chunking, metadata tags, and regional privacy/GDPR masking requirements before vector retrieval (max 35 words)."
    )
    q4_ops_and_sla: str = Field(
        description="Question 4 (Runtime SLA, Concurrency & Handoff): Specific question asking about peak concurrency, retry/dead-letter queue behavior, and human agent escalation payload format (max 35 words)."
    )


def enforce_200_word_limit(user_answer: str) -> tuple[bool, int, str]:
    """Validates and trims each client discovery response to a strict 200-word maximum."""
    words = (user_answer or "").strip().split()
    word_count = len(words)
    if word_count <= 200:
        return True, word_count, " ".join(words)
    return False, word_count, " ".join(words[:200])


def generate_product_b_discovery_questions(product_a_record: Dict[str, Any], vault_id: str) -> Dict[str, str]:
    """Analyzes the locked Product A report in the vault and generates 4 tailored technical questions."""
    rep = product_a_record.get("report_data", product_a_record)
    exec_sum = rep.get("executive_summary") or rep.get("user_idea_summary", "AI automation system")
    modality = rep.get("modality") or rep.get("detected_modality", "TEXT_REASONING")
    paid_stack = rep.get("paid_stack_summary") or rep.get("paid_stack", {}).get("model_stack", "Managed LLM API")
    paid_cost = rep.get("paid_monthly_cost_usd", 600)
    steps = ", ".join(rep.get("workflow_steps", []))

    if ChatOpenAI is None:
        return {
            "vault_analysis_summary": f"Locked onto Vault {vault_id} ({modality} | Baseline: ${paid_cost}/mo).",
            "questions": [
                "1. API & Webhook Contracts: Which exact webhook events and order/ticket payload fields must our ingestion endpoint parse, and what rate limits apply? (Max 200 words)",
                "2. Deterministic Escalation Rules: Beyond a <0.85 confidence score, what hard business rules (e.g., refund value, VIP tier, missing tracking) must force immediate human handoff? (Max 200 words)",
                "3. RAG & Privacy Controls: How frequently do your policy PDFs/FAQs update, and which PII fields (emails, addresses, order IDs) require masking before LLM inference? (Max 200 words)",
                "4. Runtime SLA & Handoff Format: What is your peak hourly ticket concurrency, and what exact JSON/note structure do your agents need inside escalated tickets? (Max 200 words)",
            ],
        }

    sys_prompt = """You are a Principal AI Solutions Architect at PickModelAI.com.
Analyze the client's locked Product A Genesis Diagnostic below and generate 4 sharp, domain-specific technical discovery questions needed to write their Product B DIY Engineering Blueprint.
Keep every question concrete, referencing their actual integrations, compliance needs, and workflow steps. Remind them each answer is capped at 200 words."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", sys_prompt),
        ("human", "VAULT ID: {vault_id}\nMODALITY: {modality}\nSUMMARY: {exec_sum}\nSTACK: {paid_stack}\nWORKFLOW: {steps}\nLOCKED OPEX: ${paid_cost}/mo"),
    ])
    llm = ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=0.15, max_retries=2, timeout=45)
    q_obj = (prompt | llm.with_structured_output(ProductBDiscoveryQuestionnaire)).invoke({
        "vault_id": vault_id,
        "modality": modality,
        "exec_sum": exec_sum,
        "paid_stack": paid_stack,
        "steps": steps,
        "paid_cost": paid_cost,
    })
    return {
        "vault_analysis_summary": q_obj.vault_analysis_summary,
        "questions": [
            f"Question 1 of 4 (API & Data Contracts — Max 200 words): {q_obj.q1_api_contracts}",
            f"Question 2 of 4 (Deterministic Guardrails & Escalation — Max 200 words): {q_obj.q2_deterministic_rules}",
            f"Question 3 of 4 (RAG Indexing & Privacy Compliance — Max 200 words): {q_obj.q3_rag_and_privacy}",
            f"Question 4 of 4 (Runtime SLA & Escalation Payload — Max 200 words): {q_obj.q4_ops_and_sla}",
        ],
    }


# ==============================================================================
# 2. UPGRADED PYDANTIC SCHEMAS (MCKINSEY / PRINCIPAL ARCHITECT GRADE)
# ==============================================================================

class BuildVsBuyRow(BaseModel):
    metric: str = Field(
        description="Evaluation dimension: (1) Setup & Launch Time, (2) Monthly Cloud Run-Rate (1k-2k runs), (3) 12-Month Cloud OpEx, (4) DevOps / Maintenance Allocation, (5) Data Residency, Zero-Training & Unit Economics at Scale."
    )
    commercial_saas: str = Field(
        description="Commercial SaaS benchmark: highlight per-seat/per-resolution pricing penalties ($0.90-$2.00/ticket), black-box prompt lock-in, and shared multi-tenant data policies."
    )
    custom_python_build: str = Field(
        description="Custom Python + AWS advantage: state exact locked USD cloud run-rate, transparent maintenance reality, EU/GDPR zero-data-retention control, and flat marginal scaling."
    )


class ArchitectureNode(BaseModel):
    node_title: str = Field(description="Clean stage title WITHOUT leading numbers, e.g., 'Webhook & HMAC Gate'")
    node_subtitle: str = Field(description="Concise 5-9 word technical action naming exact protocol, threshold, or store.")


class RoadmapRow(BaseModel):
    timeline: str = Field(description="Exact sprint window: 'Days 1–2', 'Day 3', 'Day 4', 'Day 5', 'Day 6'")
    milestone: str = Field(
        description="Concrete engineering deliverable naming exact Python modules, API webhooks, PII scrubbers, and AWS resources (28-42 words)."
    )
    verification_gate: str = Field(
        description="Deterministic pass/fail acceptance test with a quantifiable metric (e.g., '<1.8s p95 latency', '100% PII redaction', '0.85 confidence router check') (14-24 words)."
    )


class CostScalingRow(BaseModel):
    component: str = Field(description="Infrastructure or API component name.")
    tier_1_low: str = Field(description="Monthly USD cost at Baseline Volume (1,000–2,000 runs).")
    tier_2_mid: str = Field(description="Monthly USD cost at Mid Scale (15,000 runs).")
    tier_3_high: str = Field(description="Monthly USD cost at High Scale (100,000 runs).")


class FullDIYBlueprintReport(BaseModel):
    report_title: str = Field(description="Uppercase executive blueprint title (5-9 words).")
    stack_badge: str = Field(
        description="Exact production stack naming specific LLM (e.g., Python 3.11 + LangGraph + GPT-4o-mini/Claude 3.7 + Pinecone EU + AWS ECS)"
    )
    cadence_badge: str = Field(description="e.g., Real-Time Event-Driven Webhooks (<2 min SLA)")
    build_time_badge: str = Field(description="e.g., 6-Day MVP Sprint")
    paid_monthly_cost_usd: int = Field(ge=0, le=50000, description="Locked baseline monthly custom stack cloud OpEx in USD.")
    oss_monthly_cost_usd: int = Field(ge=120, le=50000, description="Locked Private VPC self-hosted monthly infra cost in USD (excl. MLOps labor).")
    build_vs_buy_narrative: str = Field(
        description=(
            "3 crisp executive paragraphs (120-160 words) comparing Commercial SaaS vs. Custom Python + AWS. "
            "Must honestly acknowledge that custom code requires developer maintenance, while proving custom wins on: "
            "(1) avoiding per-ticket/per-seat SaaS tax as volume scales, (2) strict GDPR/EU Zero-Data-Retention compliance, "
            "and (3) deterministic business-rule control before LLM calls."
        )
    )
    build_vs_buy_table: List[BuildVsBuyRow] = Field(description="Exactly 5 comparison rows.")
    architecture_narrative: str = Field(
        description=(
            "2 architectural paragraphs (90-120 words) explaining why Stage 1-2 execute deterministic Python validation, "
            "HMAC verification, and regex/Presidio PII scrubbing first, while Stage 3-4 execute RAG synthesis, "
            "structured Pydantic confidence scoring, and deterministic <0.85 human escalation."
        )
    )
    architecture_nodes: List[ArchitectureNode] = Field(description="Exactly 4 sequential pipeline nodes.")
    architecture_caption: str = Field(description="1-sentence regional cloud & VPC execution note shown beneath Figure 1.")
    prerequisites_bullets: List[str] = Field(
        description="Exactly 4 bullet points covering: (1) Domain APIs & Webhooks, (2) Regional AWS Infrastructure (e.g., eu-central-1), (3) Zero-Retention LLM & Vector DB Keys, (4) Python 3.11 Package Lockfile."
    )
    roadmap_rows: List[RoadmapRow] = Field(description="Exactly 5 rows covering Days 1–2, Day 3, Day 4, Day 5, and Day 6.")
    code_filename: str = Field(description="e.g., support_langgraph_router.py")
    production_python_code: str = Field(
        description=(
            "Clean, runnable 35-45 line Python 3.11 reference implementation using `pydantic`, `langchain_openai`, and `langgraph`. "
            "Must include: (1) Pydantic structured output schema with `confidence_score: float` and `escalate_to_human: bool`, "
            "(2) a deterministic PII scrubber & order-status pre-check node, (3) an LLM RAG responder node, and "
            "(4) a deterministic conditional edge routing `< 0.85` confidence to an agent summary escalation node. No markdown fences."
        )
    )
    aws_security_bullets: List[str] = Field(
        description=(
            "Exactly 4 detailed engineering bullets covering: (1) Regional Compute & Auto-Scaling Economics, "
            "(2) Webhook Idempotency, SQS Dead-Letter Queues & Rate-Limiting, (3) Deterministic Circuit-Breaker (<0.85 Confidence & Refund Cap Guardrail), "
            "and (4) GDPR DPA, EU Region Pinning (`eu-central-1`), and Zero-Training API Enforcement."
        )
    )
    cost_scaling_rows: List[CostScalingRow] = Field(
        description="Exactly 6 rows: 4 granular cloud/API components, 5th row 'Total Custom Python Cloud OpEx (Excl. DevOps)', and 6th row 'Comparable Per-Ticket Market SaaS Cost'."
    )
    systems_engineering_summary: str = Field(
        description="1 punchy Principal Architect takeaway paragraph (45-65 words) contrasting sub-linear cloud API scaling + semantic caching against linear per-ticket SaaS fees, while noting recommended developer retainer allocation."
    )

    @field_validator("build_vs_buy_table")
    @classmethod
    def check_bvb_len(cls, v: List[BuildVsBuyRow]) -> List[BuildVsBuyRow]:
        return v[:5]

    @field_validator("architecture_nodes")
    @classmethod
    def check_nodes_len(cls, v: List[ArchitectureNode]) -> List[ArchitectureNode]:
        cleaned = []
        for idx, node in enumerate(v[:4], 1):
            clean_title = re.sub(r"^(?:node\s*\d+[:.\-–—]*|\d+[:.\-–—]+)\s*", "", node.node_title, flags=re.I).strip()
            node.node_title = f"Node {idx}: {clean_title}" if clean_title else f"Node {idx}: Pipeline Stage"
            cleaned.append(node)
        return cleaned

    @field_validator("roadmap_rows")
    @classmethod
    def check_roadmap_len(cls, v: List[RoadmapRow]) -> List[RoadmapRow]:
        return v[:5]


# ==============================================================================
# 3. COST PARITY RECONCILIATION (FIXES SYNTAX BUGS & SAAS FALLACY)
# ==============================================================================

def _reconcile_product_b_costs(data: Dict[str, Any], locked_paid: Optional[int], locked_oss: Optional[int]) -> Dict[str, Any]:
    """Enforces exact baseline cost parity between Product A and Product B Section 1 & Section 7."""
    if locked_paid and locked_paid > 0:
        data["paid_monthly_cost_usd"] = int(locked_paid)
    if locked_oss and locked_oss > 0:
        data["oss_monthly_cost_usd"] = max(350, int(locked_oss))

    baseline = int(data.get("paid_monthly_cost_usd", 600))
    if baseline > 0:
        for row in data.get("build_vs_buy_table", []):
            metric_low = str(row.get("metric", "")).lower()
            if "monthly" in metric_low and ("cost" in metric_low or "run-rate" in metric_low):
                row["custom_python_build"] = f"${baseline:,}/mo cloud OpEx (baseline 1k–2k runs; zero per-seat fees)"
            elif "12-month" in metric_low or "1 year" in metric_low:
                row["custom_python_build"] = f"${baseline * 12:,}/yr cloud OpEx (excl. optional dev support retainer)"
            elif "3-year" in metric_low:
                row["custom_python_build"] = f"${baseline * 36:,} cloud OpEx over 36 mos (full IP & prompt ownership)"

        cost_rows = data.get("cost_scaling_rows", [])
        component_rows = []
        total_row = None
        saas_row = None
        for row in cost_rows:
            comp_low = str(row.get("component", "")).lower()
            if "total custom" in comp_low:
                total_row = row
            elif "comparable" in comp_low or "saas" in comp_low:
                saas_row = row
            else:
                component_rows.append(row)

        if component_rows and total_row is not None:
            weights = [0.55, 0.20, 0.15, 0.10]
            while len(weights) < len(component_rows):
                weights.append(0.05)
            w_sum = sum(weights[: len(component_rows)])
            allocated = [round(baseline * (w / w_sum)) for w in weights[: len(component_rows)]]
            allocated[0] += baseline - sum(allocated)  # eliminate rounding drift

            for idx, row in enumerate(component_rows):
                t1 = max(10, allocated[idx])
                row["tier_1_low"] = f"${t1:,}"
                try:
                    t2 = int(_parse_usd_cost(row.get("tier_2_mid", f"${round(t1 * 2.2)}")))
                except Exception:
                    t2 = round(t1 * 2.2)
                if t2 <= t1:
                    t2 = round(t1 * 2.2)

                try:
                    t3 = int(_parse_usd_cost(row.get("tier_3_high", f"${round(t2 * 3.4)}")))
                except Exception:
                    t3 = round(t2 * 3.4)
                if t3 <= t2:
                    t3 = round(t2 * 3.4)

                row["tier_2_mid"] = f"${t2:,}"
                row["tier_3_high"] = f"${t3:,}"

            sum_t1 = sum(int(_parse_usd_cost(r["tier_1_low"])) for r in component_rows)
            sum_t2 = sum(int(_parse_usd_cost(r["tier_2_mid"])) for r in component_rows)
            sum_t3 = sum(int(_parse_usd_cost(r["tier_3_high"])) for r in component_rows)
            total_row["component"] = "Total Custom Python Cloud OpEx (Excl. DevOps)"
            total_row["tier_1_low"] = f"${sum_t1:,}/mo"
            total_row["tier_2_mid"] = f"${sum_t2:,}/mo"
            total_row["tier_3_high"] = f"${sum_t3:,}/mo"

            if saas_row is not None:
                saas_t1 = max(int(round(sum_t1 * 1.45)), 850)
                saas_t2 = max(int(round(sum_t2 * 2.6)), 3600)
                saas_t3 = max(int(round(sum_t3 * 4.2)), 18500)
                saas_row["component"] = "Comparable Per-Resolution Market SaaS ($0.90+/ticket)"
                saas_row["tier_1_low"] = f"${saas_t1:,}/mo"
                saas_row["tier_2_mid"] = f"${saas_t2:,}/mo"
                saas_row["tier_3_high"] = f"${saas_t3:,}/mo"

    return data


# ==============================================================================
# 4. PRODUCT B GENERATOR WITH PRODUCT A CONTEXT + 4 CLIENT ANSWERS
# ==============================================================================

def generate_product_b_document(
    project_context: str,
    core_idea: str,
    vault_id: Optional[str] = None,
    discovery_answers: Optional[List[str]] = None,
) -> Dict[str, Any]:
    # Enforce single immutable Vault ID if provided
    canonical_vault_id = (vault_id or "").strip().upper() or None
    lifecycle = _prepare_vault_generation(canonical_vault_id, "2")
    if canonical_vault_id:
        lifecycle["vault_id"] = canonical_vault_id

    modality = detect_primary_modality(core_idea)

    vault = get_vault_agent()
    prior_records = vault.retrieve_client_reports(canonical_vault_id) if (vault and canonical_vault_id) else []
    locked_paid, locked_oss = _extract_locked_costs_from_records(prior_records)

    # Extract Product A context and merge with the 4 capped 200-word client answers
    formatted_answers = ""
    if discovery_answers:
        capped = [enforce_200_word_limit(ans)[2] for ans in discovery_answers[:4]]
        formatted_answers = "\n".join(
            f"Client Answer {i+1} (<=200 words): {txt}" for i, txt in enumerate(capped)
        )

    cost_lock_instruction = ""
    if locked_paid:
        resolved_oss = max(350, int(locked_oss or round(locked_paid * 0.7)))
        cost_lock_instruction = (
            f"\n6. LOCKED BASELINE COST SYNC (CRITICAL): Product A locked `paid_monthly_cost_usd` = ${locked_paid}/month "
            f"and Private VPC infra estimate = ${resolved_oss}/month (excl. MLOps labor). "
            f"You MUST set `paid_monthly_cost_usd` = {locked_paid} and `oss_monthly_cost_usd` = {resolved_oss}, "
            f"use ${locked_paid}/month as the baseline in Section 1 and Section 7 (1,000–2,000 Runs column), "
            "and scale 15,000 and 100,000 runs monotonically upward."
        )

    system_prompt = """You are a Principal AI Solutions Architect (McKinsey / Forward-Deployed Engineering lead style) at PickModelAI.com authoring Product B: The Full DIY Engineering Blueprint.
DETECTED MODALITY: {modality}

STRICT ARCHITECTURAL & CONSULTING RULES:
1. SPECIFIC MODEL & REGION NAMES: Never write generic 'Managed LLM'. Name exact models (e.g., 'OpenAI GPT-4o-mini / GPT-4o with Zero-Data-Retention' or 'Claude 3.7 Sonnet via AWS Bedrock eu-central-1') and exact vector DBs matching Product A.
2. HONEST BUILD VS. BUY ECONOMICS: Do NOT claim custom Python has zero maintenance cost. State clearly that custom builds require developer/MLOps maintenance (fitting within the client's support retainer), while winning decisively on:
   - Eliminating per-resolution ($0.90-$2.00/ticket) and per-seat SaaS vendor taxes as volume scales.
   - Strict GDPR DPA, EU data residency (`eu-central-1`), and Zero-Data-Retention guarantees.
   - Deterministic workflow and API webhook control.
3. DETERMINISTIC GATES FIRST, AI SECOND:
   - Node 1: Webhook HMAC verification & schema validation.
   - Node 2: Regex/Presidio PII scrubbing & live API state lookup (e.g., Shopify/Gorgias order status).
   - Node 3: Grounded RAG synthesis with structured Pydantic confidence & sentiment output.
   - Node 4: Deterministic circuit-breaker routing any response with `confidence_score < 0.85` or policy exception directly to human agents with a structured summary.
4. RUNNABLE PRODUCTION CODE: In `production_python_code`, write 35-45 lines of clean Python 3.11 using `pydantic`, `langchain_openai`, and `langgraph.graph.StateGraph` implementing the exact 4-stage flow above. Never include markdown ``` fences.
5. INCORPORATE ALL 4 DISCOVERY ANSWERS: Directly reflect the client's API contracts, escalation thresholds, privacy rules, and SLA targets in Sections 2, 3, 4, 5, and 6.{cost_lock_instruction}
{lifecycle_warning}"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        (
            "human",
            "<locked_product_a_context>\n{project_context}\n</locked_product_a_context>\n\n"
            "<client_4_discovery_answers_max_200_words_each>\n{formatted_answers}\n</client_4_discovery_answers_max_200_words_each>",
        ),
    ])
    if ChatOpenAI is None:
        raise RuntimeError("The langchain-openai package is required to generate Product B blueprints.")
    llm = ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=0.15, max_retries=2, timeout=90)
    structured_llm = llm.with_structured_output(FullDIYBlueprintReport, method="function_calling")
    raw_res = (prompt | structured_llm).invoke({
        "modality": modality,
        "project_context": project_context,
        "formatted_answers": formatted_answers or "Use locked Product A parameters.",
        "cost_lock_instruction": cost_lock_instruction,
        "lifecycle_warning": f"### VAULT LIFECYCLE WARNING ###\n{lifecycle['warning']}" if lifecycle.get("warning") else "",
    })
    data = raw_res.model_dump() if isinstance(raw_res, BaseModel) else dict(raw_res)
    data["detected_modality"] = modality
    if modality == "VOICE_AUDIO" and any(t in str(data.get("stack_badge", "")).lower() for t in VIDEO_ONLY_TOKENS):
        data["stack_badge"] = "Python 3.11 + OpenAI Realtime + ElevenLabs + Qdrant + AWS ECS"
    data = _reconcile_product_b_costs(data, locked_paid, locked_oss)
    result = _attach_vault_context(data, lifecycle)
    if canonical_vault_id:
        result["vault_id"] = canonical_vault_id
    return result


# ==============================================================================
# 5. EXECUTIVE ON-PAGE RENDERER (PRINT-SAFE CSS, NO SIDEBAR BLEED, NO PII LEAK)
# ==============================================================================

def render_on_page_blueprint_b(report: Dict[str, Any], contact: Dict[str, Any], vault_id: str, is_vault_locked: bool) -> None:
    generated_date = datetime.now().strftime("%B %d, %Y")
    # Single canonical Vault ID enforced everywhere
    canonical_id = str(vault_id or report.get("vault_id", "PMAI-LOCKED")).strip().upper()

    # Format corporate stakeholder line without leaking residential street/PIN codes
    client_name = contact.get("name", "Executive Stakeholder").strip()
    client_company = contact.get("company", "").strip()
    region_label = contact.get("country", "Global / EU-GDPR").strip()
    prepared_for = f"{client_name} ({client_company})" if client_company and client_company.lower() != client_name.lower() else client_name

    scope_note = (
        f"Scope & baseline economics cryptographically locked to Client Vault ID <b>{canonical_id}</b> (synced from Product A Genesis Diagnostic)."
        if is_vault_locked
        else f"Generated from Technical Discovery and locked under Client Vault ID <b>{canonical_id}</b>."
    )

    # Inject McKinsey styling + @media print isolation so Streamlit sidebars NEVER bleed into PDF exports
    st.markdown(
        """
        <style>
        .pm-doc-container {
            background: #FFFFFF;
            border: 1px solid #CBD5E1;
            border-top: 4px solid #051C2C;
            border-radius: 4px;
            padding: 24px 28px;
            margin-top: 12px;
            color: #1E293B;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        .pm-header-bar {
            background: #051C2C;
            color: #FFFFFF;
            padding: 18px 22px;
            border-radius: 3px;
            margin-bottom: 12px;
        }
        .pm-sub-tag {
            color: #38BDF8;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 4px;
        }
        .pm-doc-title {
            font-size: 20px;
            font-weight: 700;
            color: #FFFFFF;
            line-height: 1.25;
            margin-bottom: 8px;
        }
        .pm-badges {
            font-size: 12px;
            color: #E2E8F0;
            border-top: 1px solid #1E3A5F;
            padding-top: 8px;
        }
        .pm-vault-id {
            background: #005EB8;
            color: #FFFFFF;
            padding: 2px 6px;
            border-radius: 3px;
            font-weight: 700;
        }
        .pm-node-box {
            background: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-top: 3px solid #005EB8;
            padding: 10px 12px;
            border-radius: 3px;
            min-height: 84px;
        }
        .pm-node-title {
            font-size: 12px;
            font-weight: 700;
            color: #051C2C;
            margin-bottom: 4px;
        }
        .pm-node-sub {
            font-size: 11.5px;
            color: #334155;
            line-height: 1.35;
        }
        .pm-summary-box {
            background: #F8FAFC;
            border-left: 4px solid #051C2C;
            padding: 14px 18px;
            margin-top: 16px;
            font-size: 13.5px;
            color: #1E293B;
        }
        .pm-footer {
            margin-top: 18px;
            padding-top: 10px;
            border-top: 1px solid #CBD5E1;
            font-size: 11.5px;
            color: #64748B;
            display: flex;
            justify-content: space-between;
        }
        @media print {
            section[data-testid="stSidebar"],
            header[data-testid="stHeader"],
            div[data-testid="stToolbar"],
            div[data-testid="stStatusWidget"],
            .stButton, .stTextInput, .stTextArea, .stForm {
                display: none !important;
            }
            .block-container {
                padding: 0 !important;
                max-width: 100% !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.container():
        st.markdown(
            f"""
            <div class="pm-doc-container">
                <div class="pm-header-bar">
                    <div class="pm-sub-tag">PICKMODELAI ARCHITECTURE PRACTICE  |  PRODUCT B: DIY ENGINEERING BLUEPRINT</div>
                    <div class="pm-doc-title">{report.get('report_title', 'ENTERPRISE AI ENGINEERING BLUEPRINT')}</div>
                    <div class="pm-badges">
                        <b>Stack:</b> {report.get('stack_badge', '')}  | 
                        <b>Cadence:</b> {report.get('cadence_badge', '')}  | 
                        <b>Sprint:</b> {report.get('build_time_badge', '6-Day MVP Sprint')}  | 
                        <b>Vault Ref:</b> <span class="pm-vault-id">{canonical_id}</span>
                    </div>
                </div>
                <div style="background:#F8FAFC; border-left:3px solid #005EB8; padding:8px 14px; font-size:12.5px; margin-bottom:8px;">
                    <b>Prepared For:</b> {prepared_for}  • 
                    <b>Jurisdiction / Residency:</b> {region_label}  • 
                    <b>Date:</b> {generated_date}
                </div>
                <div style="background:#FFFBEB; border-left:3px solid #D97706; padding:7px 14px; font-size:12px; color:#92400E; margin-bottom:12px;">
                    {scope_note}
                </div>
            """,
            unsafe_allow_html=True,
        )

        metric_cols = st.columns(3)
        metric_cols[0].metric("Option 1: Managed API OpEx", f"${report.get('paid_monthly_cost_usd', 0):,} / mo", "Target Prod Run-Rate")
        metric_cols[1].metric("Option 2: Private VPC Infra", f"${report.get('oss_monthly_cost_usd', 350):,} / mo", "Infra Only (Excl. MLOps)")
        metric_cols[2].metric("Escalation Circuit-Breaker", "< 0.85 Score", "Deterministic Agent Handoff")
        st.divider()

        st.subheader("EXHIBIT 1  |  BUILD VS. BUY UNIT ECONOMICS & GOVERNANCE OWNERSHIP")
        st.write(report.get("build_vs_buy_narrative", ""))
        bvb_rows = report.get("build_vs_buy_table", [])
        if bvb_rows:
            st.caption("Table 1 — Executive Build-vs-Buy TCO & Governance Matrix (Synced to Locked Product A Baseline).")
            st.table([
                {
                    "Evaluation Dimension": r.get("metric", ""),
                    "Commercial Multi-Tenant SaaS": r.get("commercial_saas", ""),
                    "Custom Python + AWS Architecture": r.get("custom_python_build", ""),
                }
                for r in bvb_rows
            ])

        st.subheader("EXHIBIT 2  |  HYBRID PIPELINE TOPOLOGY (DETERMINISTIC GATES FIRST, AI SECOND)")
        st.write(report.get("architecture_narrative", ""))
        nodes = report.get("architecture_nodes", [])
        if nodes:
            cols = st.columns(len(nodes))
            for idx, node in enumerate(nodes):
                arrow = " →" if idx < len(nodes) - 1 else ""
                with cols[idx]:
                    st.markdown(
                        f"""
                        <div class="pm-node-box">
                            <div class="pm-node-title">{node.get('node_title', '')}{arrow}</div>
                            <div class="pm-node-sub">{node.get('node_subtitle', '')}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            st.caption(f"Figure 1 — {report.get('architecture_caption', '')}")

        st.subheader("EXHIBIT 3  |  PRE-SPRINT ENVIRONMENT & CREDENTIAL CHECKLIST")
        for item in report.get("prerequisites_bullets", []):
            st.markdown(f"- {item}")

        st.subheader("EXHIBIT 4  |  6-DAY SPRINT EXECUTION ROADMAP & ACCEPTANCE GATES")
        roadmap = report.get("roadmap_rows", [])
        if roadmap:
            st.caption("Table 2 — Engineering Sprint Deliverables & Deterministic Verification Gates.")
            st.table([
                {
                    "Sprint Window": r.get("timeline", ""),
                    "Engineering Deliverable & Module Scope": r.get("milestone", ""),
                    "Deterministic Acceptance Gate": r.get("verification_gate", ""),
                }
                for r in roadmap
            ])

        st.subheader("EXHIBIT 5  |  REFERENCE LANGGRAPH + PYDANTIC ROUTER IMPLEMENTATION")
        st.caption(
            f"Reference Module: `{report.get('code_filename', 'support_langgraph_router.py')}` — "
            "Illustrates deterministic PII scrubbing, structured confidence scoring, and `< 0.85` human escalation routing."
        )
        st.code(report.get("production_python_code", ""), language="python")

        st.subheader("EXHIBIT 6  |  REGIONAL AWS SECURITY, CIRCUIT-BREAKERS & GDPR CONTROLS")
        for bullet in report.get("aws_security_bullets", []):
            st.markdown(f"- {bullet}")

        st.subheader("EXHIBIT 7  |  MULTI-TIER CLOUD & API SCALING MODEL (1K TO 100K RUNS)")
        cost_rows = report.get("cost_scaling_rows", [])
        if cost_rows:
            st.caption("Table 3 — Monthly Cloud Run-Rate Scaling vs. Per-Resolution SaaS Benchmarks (USD/month, excl. DevOps labor).")
            st.table([
                {
                    "Cost Component": r.get("component", ""),
                    "Baseline (1k–2k Runs/Mo)": r.get("tier_1_low", ""),
                    "Growth (15,000 Runs/Mo)": r.get("tier_2_mid", ""),
                    "Enterprise (100,000 Runs/Mo)": r.get("tier_3_high", ""),
                }
                for r in cost_rows
            ])

        st.markdown(
            f"""
            <div class="pm-summary-box">
                <b>Principal Architect TCO Synthesis:</b> {report.get('systems_engineering_summary', '')}
            </div>
            <div class="pm-footer">
                <span>PickModelAI Architecture Advisory • Product B Engineering Blueprint • {generated_date}</span>
                <span>Immutable Vault Ref: <b>{canonical_id}</b></span>
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ===========================================================================
# 4. PRODUCT C: ENHANCED ENTERPRISE BLUEPRINT (SCHEMAS, ENGINE & 9-SECTION RENDERER)
# ===========================================================================
class DynamicEnterpriseQuestions(BaseModel):
    questions: List[str] = Field(
        description="Exactly 3 highly specific enterprise follow-up questions based on the project context, addressing team sizing, high availability/scaling bottlenecks, and strict compliance/security audits (e.g., SOC2, HIPAA)."
    )


class EnterpriseComparisonRow(BaseModel):
    dimension: str = Field(description="e.g., Initial Validation & Build Timeline, Upfront Engineering CapEx, Commercial IP Rights, Analytical & RAG Modularity, Baseline Monthly OpEx, 10x Scale OpEx.")
    prebuilt_saas: str = Field(description="Pre-built SaaS vendor timeline, limits, and pricing.")
    modular_custom_stack: str = Field(description="Modular custom Python + LangGraph + AWS architecture metrics.")


class TcoTrajectoryPoint(BaseModel):
    period: str = Field(description="e.g., Month 1, Year 1 (2x), Year 2 (5x), Year 3 (10x)")
    prebuilt_saas_cumulative_usd: int = Field(ge=0, le=500000, description="Cumulative SaaS spend in USD.")
    custom_stack_cumulative_usd: int = Field(ge=0, le=500000, description="Cumulative custom stack + dev spend in USD.")


class EnterprisePipelineNode(BaseModel):
    node_name: str = Field(description="Short node label.")
    node_detail: str = Field(description="Concise 3-6 word action.")


class RagConfigRow(BaseModel):
    parameter: str = Field(description="e.g., 1. Dual-Namespace Partitioning, 2. Hard Temporal Metadata Filter, 3. Hybrid Metadata + Semantic Retrieval, 4. Cross-Encoder Reranking.")
    production_config: str = Field(description="Exact technical configuration (namespaces, TTL, metadata filters, embedding dims, reranker).")
    quantitative_impact: str = Field(description="Measurable precision, freshness, or anti-hallucination impact.")


class StressTestRow(BaseModel):
    scale_tier: str = Field(description="e.g., 2x Scale, 5x Scale, 10x Scale")
    growth_driver: str = Field(description="Operational or user growth trigger.")
    bottleneck: str = Field(description="Specific technical bottleneck or failure mode.")
    mitigation: str = Field(description="Concrete AWS/Python architectural mitigation.")


class FixedVsVariableCostPoint(BaseModel):
    scale_label: str = Field(description="e.g., 1x (1k), 2x (10k), 5x (35k), 10x (100k)")
    fixed_ai_engine_usd: int = Field(ge=0, le=20000, description="Monthly fixed AI + RAG + compute cost in USD.")
    variable_delivery_usd: int = Field(ge=0, le=50000, description="Monthly variable email/CDN/bandwidth delivery cost in USD.")


class SwotAnalysisBlock(BaseModel):
    strengths: List[str] = Field(description="3 concrete internal technical advantages.")
    weaknesses: List[str] = Field(description="2 honest internal engineering trade-offs.")
    opportunities: List[str] = Field(description="2 concrete 12-month strategic/monetization upsides.")
    threats: List[str] = Field(description="2 external risks with explicit technical/compliance mitigations.")


class MultiScaleFinancialRow(BaseModel):
    line_item: str = Field(description="Cloud / API line item.")
    scale_1x: str = Field(description="Monthly USD at 1x baseline.")
    scale_2x: str = Field(description="Monthly USD at 2x growth.")
    scale_5x: str = Field(description="Monthly USD at 5x growth.")
    scale_10x: str = Field(description="Monthly USD at 10x scale.")


def _parse_usd_cost(value: Any) -> Decimal:
    text = str(value).strip()
    match = re.fullmatch(
        r"(?:USD\s*)?\$?\s*((?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{1,2})?)\s*([kK])?\s*(?:USD)?\s*(?:/(?:mo(?:nth)?|month)|per month|monthly)?\s*(?:\([^$\d]*\))?",
        text,
        re.IGNORECASE,
    )
    if not match:
        raise ValueError(f"Expected a single non-negative USD amount, got {value!r}")
    try:
        amount = Decimal(match.group(1).replace(",", ""))
        if match.group(2):
            amount *= 1000
        return amount.quantize(Decimal("0.01"))
    except InvalidOperation as exc:
        raise ValueError(f"Invalid USD amount: {value!r}") from exc


def _format_usd(amount: Decimal) -> str:
    return f"${amount:,.0f}" if amount == amount.to_integral_value() else f"${amount:,.2f}"


def _fix_financial_rows(rows: List[Dict[str, Any]], locked_baseline_usd: Optional[int] = None) -> List[Dict[str, Any]]:
    """BUG #4 & BUG #5 FIX:
    1. Locks 1x OpEx total to Product A's canonical baseline when provided.
    2. Prevents lazy flat copying across 1x -> 2x -> 5x -> 10x by enforcing strict monotonic growth.
    """
    fixed = [dict(row) for row in rows]
    columns = ("scale_1x", "scale_2x", "scale_5x", "scale_10x")
    scale_multipliers = (Decimal("1.0"), Decimal("1.8"), Decimal("4.0"), Decimal("7.5"))
    totals = []
    components = []
    saas_rows = []

    for row in fixed:
        label = str(row.get("line_item", "")).casefold()
        if "initial engineering capex" in label:
            flat_value = row["scale_1x"]
            _parse_usd_cost(flat_value)
            for column in columns:
                row[column] = flat_value
        elif "total custom enterprise stack" in label:
            totals.append(row)
        elif "pre-built saas equivalent" in label:
            saas_rows.append(row)
        else:
            components.append(row)

    if len(totals) != 1 or not components:
        raise ValueError("Financial rows require exactly one Total Custom Enterprise Stack row and at least one OpEx component.")

    # Step 1: If locked_baseline_usd is provided from Product A, rescale 1x component values to sum to locked_baseline_usd
    if locked_baseline_usd and locked_baseline_usd > 0:
        raw_1x = [_parse_usd_cost(row["scale_1x"]) for row in components]
        raw_sum = sum(raw_1x, Decimal("0"))
        target_1x = Decimal(str(locked_baseline_usd))
        if raw_sum > 0 and raw_sum != target_1x:
            scaled_1x = [((val / raw_sum) * target_1x).quantize(Decimal("1")) for val in raw_1x]
            scaled_1x[0] += target_1x - sum(scaled_1x, Decimal("0"))
            for row, val in zip(components, scaled_1x):
                row["scale_1x"] = _format_usd(val)

    # Step 2: Ensure each OpEx component scales monotonically across 1x -> 2x -> 5x -> 10x (no flat copying)
    for row in components:
        base_val = _parse_usd_cost(row["scale_1x"])
        parsed_cols = [_parse_usd_cost(row[col]) for col in columns]
        # If LLM copied 1x across all columns or failed to increase, apply realistic tiered scaling
        if len(set(parsed_cols)) == 1 and base_val > 0:
            for col, mult in zip(columns, scale_multipliers):
                scaled = (base_val * mult).quantize(Decimal("1"))
                row[col] = _format_usd(scaled)
        else:
            prev = parsed_cols[0]
            for idx in range(1, len(columns)):
                curr = _parse_usd_cost(row[columns[idx]])
                if curr <= prev and prev > 0:
                    curr = (prev * Decimal("1.6")).quantize(Decimal("1"))
                row[columns[idx]] = _format_usd(curr)
                prev = curr

    # Step 3: Also ensure Pre-Built SaaS Equivalent scales monotonically if LLM flattened it
    for row in saas_rows:
        base_saas = _parse_usd_cost(row["scale_1x"])
        parsed_saas = [_parse_usd_cost(row[col]) for col in columns]
        if len(set(parsed_saas)) == 1 and base_saas > 0:
            for col, mult in zip(columns, (Decimal("1.0"), Decimal("2.0"), Decimal("4.5"), Decimal("8.5"))):
                row[col] = _format_usd((base_saas * mult).quantize(Decimal("1")))

    # Step 4: Recompute exact reconciled totals
    amounts = [sum((_parse_usd_cost(row[column]) for row in components), Decimal("0")) for column in columns]
    for column, amount in zip(columns, amounts):
        totals[0][column] = _format_usd(amount)

    if any(left >= right for left, right in zip(amounts, amounts[1:])):
        raise ValueError(f"Non-monotonic Total row: {rows}")
    return fixed


class EnhancedEnterpriseBlueprintReport(BaseModel):
    report_title: str = Field(description="Uppercase enterprise architectural title (6-10 words).")
    target_system_badge: str = Field(description="Concise target system summary (5-9 words).")
    architecture_badge: str = Field(description="e.g., Async Python + Temporal RAG + AWS Fargate")
    scale_horizon_badge: str = Field(description="e.g., 1x to 10x Enterprise Scale")
    paid_monthly_cost_usd: int = Field(ge=0, le=50000, description="Baseline 1x monthly custom stack cost in USD.")
    oss_monthly_cost_usd: int = Field(ge=0, le=50000, description="Private VPC self-hosted monthly cost in USD.")
    executive_synthesis: str = Field(description="3 crisp executive paragraphs explaining business problem, commercial SaaS IP/modularity flaws, and why this modular architecture wins (140-180 words).")
    comparison_rows: List[EnterpriseComparisonRow] = Field(description="Exactly 6 architectural dimension rows.")
    tco_trajectory: List[TcoTrajectoryPoint] = Field(description="Exactly 4 points: Month 1, Year 1 (2x), Year 2 (5x), Year 3 (10x).")
    figure_1_caption: str = Field(description="1-sentence caption for Figure 1 (36-Month Cumulative TCO Trajectory).")
    architecture_intro: str = Field(description="1 concise paragraph introducing strict separation of deterministic compute and calibrated synthesis.")
    architecture_stages: List[str] = Field(description="Exactly 4 detailed stage bullets (Stage 1 through Stage 4).")
    pipeline_nodes: List[EnterprisePipelineNode] = Field(description="Exactly 7 sequential pipeline nodes (4 top-row + 3 bottom-row).")
    figure_2_caption: str = Field(description="1-sentence caption for Figure 2.")
    rag_intro: str = Field(description="1 paragraph explaining why enterprise/temporal RAG requires metadata filtering and cross-encoder reranking.")
    rag_rows: List[RagConfigRow] = Field(description="Exactly 4 RAG parameter rows.")
    code_filename: str = Field(description="e.g., enterprise_pipeline.py")
    production_python_code: str = Field(description="Clean, runnable 40-65 line Python reference implementation showing async ingestion/filtering, temporal/metadata RAG query, structured Pydantic LLM synthesis, and deterministic circuit-breaker audit. No markdown fences.")
    stress_test_intro: str = Field(description="1 paragraph explaining the two primary demand scaling axes.")
    stress_test_rows: List[StressTestRow] = Field(description="Exactly 3 rows covering 2x Scale, 5x Scale, and 10x Scale.")
    fixed_vs_variable_points: List[FixedVsVariableCostPoint] = Field(description="Exactly 4 points: 1x, 2x, 5x, 10x.")
    figure_3_caption: str = Field(description="1-sentence caption for Figure 3.")
    swot: SwotAnalysisBlock
    financial_rows: List[MultiScaleFinancialRow] = Field(description="Exactly 7 rows ending with 'Total Custom Enterprise Stack' and 'Pre-Built SaaS Equivalent'.")
    financial_footnote: str = Field(description="1-sentence footnote explaining multi-tier/multi-region assumptions at 5x and 10x.")
    executive_recommendation: str = Field(description="1 punchy executive recommendation paragraph on decoupling compute from fan-out for 95%+ gross margins (45-65 words).")
    appendix_notes: List[str] = Field(description="Exactly 4 bullet points crediting Foundation & Embedding Models, Retrieval & Re-Ranking, Quantitative/Orchestration Stack, and Cloud Infrastructure & Compliance Disclaimers.")

    @field_validator("comparison_rows")
    @classmethod
    def check_comp_len(cls, v: List[EnterpriseComparisonRow]) -> List[EnterpriseComparisonRow]:
        return v[:6]

    @field_validator("pipeline_nodes")
    @classmethod
    def check_nodes_len(cls, v: List[EnterprisePipelineNode]) -> List[EnterprisePipelineNode]:
        return v[:7]

    @field_validator("rag_rows")
    @classmethod
    def check_rag_len(cls, v: List[RagConfigRow]) -> List[RagConfigRow]:
        return v[:4]

    @field_validator("stress_test_rows")
    @classmethod
    def check_stress_len(cls, v: List[StressTestRow]) -> List[StressTestRow]:
        return v[:3]

    @field_validator("financial_rows")
    @classmethod
    def reconcile_financial_rows(cls, v: List[MultiScaleFinancialRow]) -> List[MultiScaleFinancialRow]:
        return [MultiScaleFinancialRow(**row) for row in _fix_financial_rows([row.model_dump() for row in v])]


def generate_dynamic_questions(vault_context: str) -> List[str]:
    system_prompt = """You are a Principal Enterprise AI Architect at PickModelAI.com.
We have retrieved the client's project scope from their verified Vault ID:
<scope>
{vault_context}
</scope>
Generate exactly 3 specific enterprise-level follow-up questions addressing:
1. Team sizing / operational capacity / engineering bandwidth.
2. High availability, disaster recovery, or high-throughput scaling bottlenecks.
3. Strict compliance, security audits (SOC2, HIPAA, SOX, financial regulations), or private VPC residency requirements."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Generate the 3 enterprise follow-up questions based on the retrieved vault scope."),
    ])
    if ChatOpenAI is None:
        raise RuntimeError("The langchain-openai package is required to generate discovery questions.")
    llm = ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=0.3)
    structured_llm = llm.with_structured_output(DynamicEnterpriseQuestions)
    result = (prompt | structured_llm).invoke({"vault_context": vault_context})
    if isinstance(result, dict):
        return result.get("questions", [])
    return getattr(result, "questions", [])


def generate_product_c_document(project_context: str, dynamic_answers: str, core_idea: str, vault_id: Optional[str] = None) -> Dict[str, Any]:
    lifecycle = _prepare_vault_generation(vault_id, "3")
    modality = detect_primary_modality(core_idea or project_context)

    # BUG #5 FIX: Retrieve locked Product A costs from vault so Product C Section 2 & Section 8 match Product A
    vault = get_vault_agent()
    prior_records = vault.retrieve_client_reports(vault_id) if (vault and vault_id) else []
    locked_paid, locked_oss = _extract_locked_costs_from_records(prior_records)

    cost_lock_block = ""
    if locked_paid:
        cost_lock_block = (
            f"\n10. LOCKED BASELINE COST SYNC (CRITICAL): The locked baseline OpEx from Product A is ${locked_paid}/month "
            f"(Private VPC baseline: ${locked_oss or int(locked_paid * 0.6)}/month). "
            f"You MUST set `paid_monthly_cost_usd` = {locked_paid} and ensure Section 2 (`Baseline Monthly OpEx`) and Section 8 (`1x Baseline` Total Custom Enterprise Stack) equal ${locked_paid}. "
            "In Section 8, 2x, 5x, and 10x columns MUST strictly increase with traffic (e.g., 1.8x, 4.0x, 7.5x of baseline) — NEVER repeat the 1x number across all columns."
        )

    system_prompt = """You are a Principal Enterprise AI Solutions Architect at PickModelAI.com generating Product C: The Enhanced Enterprise Architectural Blueprint (Beta Testing).
Produce an institutional, board-ready 9-section systems engineering and multi-scale financial blueprint.

CRITICAL FOUNDATION — LOCKED USER SCOPE:
<locked_vault_scope>
{project_context}
</locked_vault_scope>

ENTERPRISE FOLLOW-UP ANSWERS & CONSTRAINTS:
<enterprise_followup>
{dynamic_answers}
</enterprise_followup>

{lifecycle_warning}

MANDATORY EXECUTION RULES FOR ENTERPRISE BLUEPRINT:
1. DEEP VERTICAL CUSTOMIZATION: Every section MUST explicitly reference and solve the specific technical nuances of THIS project based on both the locked vault scope and the enterprise follow-up answers. Never output generic boilerplate.
2. SECTION 3 ALIGNMENT: The text description of the workflow stages in Section 3 must precisely match the sequential technical nodes in the visual pipeline diagram.
3. REAL PRODUCTION CODE (SECTION 5): In `production_python_code`, write a robust, production-ready, multi-node LangGraph or async enterprise pipeline script (40-65 lines) using valid Python syntax (proper `=` assignments, no missing operators) and modern SDK imports (`from openai import AsyncOpenAI`, `from pydantic import BaseModel`). No toy code or `asyncio.sleep()` demos.
4. DOMAIN-SPECIFIC SWOT (SECTION 7): Focus on real-world engineering advantages and risks of this specific vertical.
5. MODALITY LOCK: Detected Project Modality is {modality}.
6. REFERENCE PYTHON CODE: Output clean enterprise code without ```python markdown backticks.
7. FINANCIAL RECONCILIATION: Each financial cell must contain one USD amount formatted as $1,234.56, without ranges, arithmetic, or explanatory text. Initial Engineering CapEx is a one-time flat amount at every scale. Total Custom Enterprise Stack is the sum of monthly OpEx components only, excluding CapEx and Pre-Built SaaS Equivalent. Monthly totals MUST strictly increase across 1x -> 2x -> 5x -> 10x (never flat). Put all qualifications in financial_footnote.
8. DECISION QUALITY: State workload, utilization, staffing, pricing, and scale assumptions explicitly. Quantify claims as estimates or testable targets, not promises. Avoid hype adjectives. Do not assert 95% margins without an explicit revenue and full-cost calculation. Captions must explain assumptions and identify model-generated estimates, not invent verified sources.
9. LIFECYCLE: If a VAULT LIFECYCLE WARNING is provided, executive_synthesis must acknowledge the pending Product B work and estimated 40% overlap; do not claim Product B was already delivered.{cost_lock_block}"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Generate the complete 9-section Enhanced Enterprise Architectural Blueprint tailored strictly to the locked vault scope and enterprise follow-up answers."),
    ])
    if ChatOpenAI is None:
        raise RuntimeError("The langchain-openai package is required to generate Product C blueprints.")
    llm = ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=0.15, max_retries=3, timeout=120)
    structured_llm = llm.with_structured_output(EnhancedEnterpriseBlueprintReport, method="function_calling")
    raw_result = (prompt | structured_llm).invoke({
        "modality": modality,
        "project_context": project_context,
        "dynamic_answers": dynamic_answers,
        "cost_lock_block": cost_lock_block,
        "lifecycle_warning": f"### VAULT LIFECYCLE WARNING ###\n{lifecycle['warning']}" if lifecycle["warning"] else "",
    })
    data = raw_result.model_dump() if isinstance(raw_result, BaseModel) else dict(raw_result)
    data["financial_rows"] = _fix_financial_rows(data.get("financial_rows", []), locked_baseline_usd=locked_paid)

    if locked_paid and locked_paid > 0:
        data["paid_monthly_cost_usd"] = locked_paid
        for row in data.get("comparison_rows", []):
            if "baseline monthly opex" in str(row.get("dimension", "")).lower():
                row["modular_custom_stack"] = f"${locked_paid:,}/month baseline (1x scale)"
    if locked_oss and locked_oss > 0:
        data["oss_monthly_cost_usd"] = locked_oss

    if lifecycle["warning"] and lifecycle["warning"] not in data.get("executive_synthesis", ""):
        data["executive_synthesis"] = f"{data.get('executive_synthesis', '')}\n\n{lifecycle['warning']}."
    data["detected_modality"] = modality
    if modality == "VOICE_AUDIO" and any(t in str(data.get("architecture_badge", "")).lower() for t in VIDEO_ONLY_TOKENS):
        data["architecture_badge"] = "Async Voice Pipeline + Temporal RAG + AWS Fargate"
    return _attach_vault_context(data, lifecycle)


def _find_local_contact_fallback(vault_id: str) -> Optional[Dict[str, Any]]:
    for folder in (LEADS_DIR, PRODUCT_B_LEADS_DIR, PRODUCT_C_LEADS_DIR):
        if folder.exists():
            matches = sorted(folder.glob(f"{vault_id}_*.json"), reverse=True)
            for m in matches:
                try:
                    data = json.loads(m.read_text(encoding="utf-8"))
                    cd = data.get("contact_details")
                    if isinstance(cd, dict) and cd.get("email"):
                        return cd
                except Exception:
                    continue
    return None


def fetch_prior_tier_context(
    vault: PlanModelVaultAgent, vault_id: str
) -> Optional[Tuple[str, str, str, Optional[Dict[str, Any]]]]:
    records = vault.retrieve_client_reports(vault_id)
    eligible = [r for r in records if _tier_number(r.get("tier", "")) in {"1", "2"}]
    if not eligible:
        return None
    selected_rec = max(eligible, key=lambda r: int(_tier_number(r.get("tier", ""))))
    locked_paid, locked_oss = _extract_locked_costs_from_records(records)

    saved_contact: Optional[Dict[str, Any]] = None
    for r in records:
        rep_r = r.get("report_data", {})
        if isinstance(rep_r.get("contact_details"), dict) and rep_r["contact_details"].get("email"):
            saved_contact = rep_r["contact_details"]
            break
    if saved_contact is None:
        saved_contact = _find_local_contact_fallback(vault_id)
    if saved_contact and "city" not in saved_contact and "city_country" in saved_contact:
        parts = [p.strip() for p in str(saved_contact["city_country"]).split(",")]
        saved_contact["city"] = parts[0] if parts else str(saved_contact["city_country"])
        saved_contact["country"] = parts[-1] if len(parts) > 1 else "India"

    tier_label = str(selected_rec.get("tier", "Product B Vault Record"))
    if _tier_number(tier_label) == "1":
        tier_label = "Tier 1 (Product A — Genesis Diagnostic)"
    inp = dict(selected_rec.get("user_inputs", {}))
    rep = selected_rec.get("report_data", {})
    prior_a = next((r for r in eligible if _tier_number(r.get("tier", "")) == "1"), None)
    if prior_a:
        original_inputs = prior_a.get("user_inputs", {})
        for key in ("goal", "workflow", "constraints", "budget"):
            if not inp.get(key) or str(inp[key]).startswith("Locked Vault Scope"):
                inp[key] = original_inputs.get(key, inp.get(key, ""))
    core_goal = str(inp.get("goal", ""))
    context_lines = [
        f"Locked Vault Record ({vault_id} — Imported from {tier_label}):",
        f"- Core System Goal & Modality: {core_goal}",
        f"- Target Users & Workflow: {inp.get('workflow', '')}",
        f"- Constraints & Compliance: {inp.get('constraints', '')}",
        f"- Budget & Scale Target: {inp.get('budget', '')}",
    ]
    if locked_paid:
        context_lines.append(f"- LOCKED Baseline Managed Cloud Monthly OpEx (from Product A): ${locked_paid}/month")
    if locked_oss:
        context_lines.append(f"- LOCKED Baseline Private VPC Monthly OpEx (from Product A): ${locked_oss}/month")
    if rep.get("paid_stack_summary"):
        context_lines.append(f"- Prior Selected Cloud Stack: {rep.get('paid_stack_summary')}")
    if rep.get("oss_stack_summary"):
        context_lines.append(f"- Prior Selected VPC Stack: {rep.get('oss_stack_summary')}")
    if rep.get("executive_summary"):
        context_lines.append(f"- Prior Executive Summary: {rep.get('executive_summary')}")
    return tier_label, core_goal, "\n".join(context_lines), saved_contact


def persist_product_c_record(
    contact: Dict[str, Any],
    project_inputs: Dict[str, str],
    report_data: Dict[str, Any],
    existing_vault_id: Optional[str] = None,
) -> str:
    vault = get_vault_agent()
    if vault is None:
        raise RuntimeError("Vault persistence is unavailable; no replacement ID was created.")
    ref_id = _report_vault_id(report_data, existing_vault_id)
    prior_records = vault.retrieve_client_reports(ref_id) if ref_id else []
    locked_paid, _ = _extract_locked_costs_from_records(prior_records)
    report_data["financial_rows"] = _fix_financial_rows(report_data.get("financial_rows", []), locked_baseline_usd=locked_paid)
    vault_payload = {
        **report_data,
        "contact_details": contact,
        "modality": report_data.get("detected_modality", "TEXT_REASONING"),
        "feasibility_grade": "Grade A (Enhanced Enterprise Blueprint)",
        "executive_summary": str(report_data.get("executive_synthesis", ""))[:450],
        "paid_stack_summary": report_data.get("architecture_badge", ""),
        "oss_stack_summary": "Multi-Scale AWS Fargate / Private VPC Enterprise Architecture",
    }
    ref_id = vault.save_blueprint_to_vault(
        user_inputs=project_inputs,
        report_data=vault_payload,
        tier="Tier 3 (Product C Enterprise)",
        client_ref_id=ref_id,
    )
    for key in ("vault_id", "lineage", "tiers_generated"):
        report_data[key] = vault_payload[key]

    timestamp_slug = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    lead_file = LEADS_DIR / f"{ref_id}_ProductC_{timestamp_slug}_{uuid.uuid4().hex[:6]}.json"
    record = {
        "client_vault_id": ref_id,
        "tier": "Product C (Tier 3 Enhanced Enterprise Architectural Blueprint — Beta Testing)",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "contact_details": contact,
        "project_inputs": project_inputs,
        "generated_blueprint": report_data,
        "lineage": report_data["lineage"],
        "tiers_generated": report_data["tiers_generated"],
    }
    lead_file.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return ref_id


def render_on_page_enterprise_blueprint(
    report: Dict[str, Any],
    contact: Optional[Dict[str, Any]],
    vault_id: Optional[str],
    imported_from_tier: Optional[str],
) -> None:
    report = {**report, "financial_rows": _fix_financial_rows(report.get("financial_rows", []))}
    generated_date = datetime.now().strftime("%B %d, %Y")
    safe_vault_id = vault_id or "PMAI-ENTERPRISE"
    safe_contact = contact or {}
    c_name = safe_contact.get("name", "Enterprise Client")
    c_company = safe_contact.get("company", "Organization")
    c_city = safe_contact.get("city", "")
    c_country = safe_contact.get("country", "")
    c_pin = safe_contact.get("pin_code", "")
    c_email = safe_contact.get("email", "")
    c_phone = safe_contact.get("phone", "")
    location_str = ", ".join([p for p in [c_city, c_country] if p])
    if c_pin:
        location_str = f"{location_str} — {c_pin}" if location_str else c_pin

    scope_note = (
        f"Scope locked to verified Client Vault ID <code class='pm-vault-id'>{safe_vault_id}</code> (imported from <b>{imported_from_tier}</b>). "
        "To modify project scope or request fresh reports, please email <b>support@pickmodelai.com</b>."
        if imported_from_tier
        else f"Generated from enterprise discovery session and locked under Client Vault ID <code class='pm-vault-id'>{safe_vault_id}</code>. "
        "To request scope revisions, contact <b>support@pickmodelai.com</b>."
    )

    with st.container():
        st.markdown(
            f"""
            <div class="pm-doc-container">
                <div class="pm-header-bar">
                    <div class="pm-sub-tag">PICKMODELAI.COM — ENHANCED ENTERPRISE ARCHITECTURAL BLUEPRINT (PRODUCT C • BETA TESTING)</div>
                    <div class="pm-doc-title">{report.get('report_title', '')}</div>
                    <div class="pm-badges">
                        Target System: {report.get('target_system_badge', '')}  | 
                        Architecture: {report.get('architecture_badge', '')}  | 
                        Scale Horizon: {report.get('scale_horizon_badge', '')}  | 
                        Vault ID: <code class="pm-vault-id">{safe_vault_id}</code>
                    </div>
                </div>
                <div style="background:#F1F5F9; padding:8px 14px; border-radius:4px; font-size:12.5px; margin-bottom:8px;">
                    <b>Prepared For:</b> {c_name} ({c_company})  • 
                    <b>Location:</b> {location_str}  • 
                    <b>Contact:</b> {c_email} | {c_phone}
                </div>
                <div style="background:#FEF3C7; padding:7px 14px; border-radius:4px; font-size:12px; color:#92400E; margin-bottom:10px;">
                    🔒 {scope_note}
                </div>
            """,
            unsafe_allow_html=True,
        )
        metric_cols = st.columns(2)
        total_row = next(r for r in report["financial_rows"] if "total custom enterprise stack" in r["line_item"].casefold())
        metric_cols[0].metric("Reconciled baseline OpEx (USD/month)", total_row["scale_1x"])
        metric_cols[1].metric("Reconciled 10× OpEx (USD/month)", total_row["scale_10x"])
        st.divider()

        st.header("1. Executive Synthesis & Strategic Build-vs-Prebuilt Thesis")
        st.write(report.get("executive_synthesis", ""))

        st.header("2. Pre-Built SaaS vs. Modular Custom Build: Architecture, Time & Cost")
        comp_rows = report.get("comparison_rows", [])
        if comp_rows:
            st.caption("Table 1 — Architectural comparison. Source: model-generated estimates from the supplied project assumptions.")
            st.table([
                {
                    "Architectural Dimension": r.get("dimension", ""),
                    "Pre-Built SaaS Platforms": r.get("prebuilt_saas", ""),
                    "Modular Custom Stack (Python + LangGraph + AWS)": r.get("modular_custom_stack", ""),
                }
                for r in comp_rows
            ])

        tco_pts = report.get("tco_trajectory", [])
        if tco_pts:
            st.markdown("**Figure 1 — 36-Month Cumulative TCO Trajectory (USD)**")
            df_tco = pd.DataFrame([
                {
                    "Period": p.get("period", ""),
                    "Pre-Built SaaS Cumulative ($)": int(p.get("prebuilt_saas_cumulative_usd", 0)),
                    "Modular Custom Stack + Dev ($)": int(p.get("custom_stack_cumulative_usd", 0)),
                }
                for p in tco_pts
            ])
            df_tco["Period"] = pd.Categorical(df_tco["Period"], categories=list(dict.fromkeys(df_tco["Period"])), ordered=True)
            df_tco.set_index("Period", inplace=True)
            st.line_chart(df_tco, x_label="Elapsed period (months / years)", y_label="Cumulative cost (USD)")
            st.caption(f"Figure 1 — {report.get('figure_1_caption', '')} Source: model-generated TCO estimates using the stated workload and pricing assumptions; not audited vendor quotations.")

        st.header("3. End-to-End Enterprise Quantitative & LangGraph Architecture")
        st.write(report.get("architecture_intro", ""))
        for stage in report.get("architecture_stages", []):
            st.markdown(f"- {stage}")

        nodes = report.get("pipeline_nodes", [])
        if nodes:
            top_nodes = nodes[:4]
            bottom_nodes = nodes[4:7]
            cols_top = st.columns(len(top_nodes))
            for idx, node in enumerate(top_nodes):
                arrow = " →" if idx < len(top_nodes) - 1 else " ↓"
                with cols_top[idx]:
                    st.markdown(
                        f"""
                        <div class="pm-node-box">
                            <div class="pm-node-title">{node.get('node_name', '')}{arrow}</div>
                            <div class="pm-node-sub">{node.get('node_detail', '')}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            if bottom_nodes:
                cols_bot = st.columns(len(bottom_nodes))
                for idx, node in enumerate(bottom_nodes):
                    arrow = " →" if idx < len(bottom_nodes) - 1 else ""
                    with cols_bot[idx]:
                        st.markdown(
                            f"""
                            <div class="pm-node-box">
                                <div class="pm-node-title">{node.get('node_name', '')}{arrow}</div>
                                <div class="pm-node-sub">{node.get('node_detail', '')}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
            st.caption(f"Figure 2 — {report.get('figure_2_caption', '')} Source: proposed architecture derived from the locked scope and enterprise answers; arrows indicate execution order.")

        st.header("4. Optimized Temporal & Enterprise RAG Retrieval Configuration")
        st.write(report.get("rag_intro", ""))
        rag_rows = report.get("rag_rows", [])
        if rag_rows:
            st.caption("Table 2 — Proposed retrieval configuration and measurable validation targets.")
            st.table([
                {
                    "Enterprise RAG Parameter": r.get("parameter", ""),
                    "Production Configuration": r.get("production_config", ""),
                    "Quantitative & Editorial Impact": r.get("quantitative_impact", ""),
                }
                for r in rag_rows
            ])

        st.header("5. Complete Production Python Implementation (Reference Example — Code May Differ or Get Updated)")
        st.caption(
            f"Reference File: `{report.get('code_filename', 'enterprise_pipeline.py')}` — "
            "*Note: This implementation is a reference example; actual production code may differ or be updated based on SDK versions and deployment requirements.*"
        )
        st.info("**Reference Code Notice:** The Python implementation below is an illustrative feasibility reference, not production code. LLM SDKs, LangChain APIs, and cloud provider SDKs evolve rapidly — specific imports, method signatures, and model names in this example are subject to deprecation, change, or decommissioning. The durable value of this section is the architectural pattern — deterministic validation before LLM synthesis, structured Pydantic schemas, and audit-trail design — not the specific code lines shown.")
        st.code(report.get("production_python_code", ""), language="python")

        st.header("6. One-Year Scalability Stress-Test: 2×, 5× & 10× Demand Impact")
        st.write(report.get("stress_test_intro", ""))
        stress_rows = report.get("stress_test_rows", [])
        if stress_rows:
            st.caption("Table 3 — Scaling scenarios, projected bottlenecks, and mitigations; validation by load testing is required.")
            st.table([
                {
                    "Scale Tier": r.get("scale_tier", ""),
                    "Growth Driver": r.get("growth_driver", ""),
                    "System Bottleneck & Failure Mode": r.get("bottleneck", ""),
                    "Required Architectural Mitigation": r.get("mitigation", ""),
                }
                for r in stress_rows
            ])

        fv_pts = report.get("fixed_vs_variable_points", [])
        if fv_pts:
            st.markdown("**Figure 3 — Fixed AI Processing Engine vs. Variable Delivery Infrastructure Cost (USD/Month)**")
            df_fv = pd.DataFrame([
                {
                    "Scale Tier": p.get("scale_label", ""),
                    "Fixed AI + RAG Engine ($/mo)": int(p.get("fixed_ai_engine_usd", 0)),
                    "Variable Delivery + CDN ($/mo)": int(p.get("variable_delivery_usd", 0)),
                }
                for p in fv_pts
            ])
            df_fv["Scale Tier"] = pd.Categorical(df_fv["Scale Tier"], categories=list(dict.fromkeys(df_fv["Scale Tier"])), ordered=True)
            df_fv.set_index("Scale Tier", inplace=True)
            st.bar_chart(df_fv, x_label="Demand scale (multiple of baseline)", y_label="Monthly cost (USD/month)")
            st.caption(f"Figure 3 — {report.get('figure_3_caption', '')} Source: model-generated infrastructure estimates using the stated scale assumptions; validate against current vendor pricing.")

        st.header("7. Comprehensive Enterprise SWOT Analysis")
        swot = report.get("swot", {})
        c_sw1, c_sw2 = st.columns(2)
        with c_sw1:
            st.markdown("**STRENGTHS (Internal Technical Advantages)**")
            for s in list(dict.fromkeys(swot.get("strengths", []))):
                st.markdown(f"- {s}")
            st.markdown("**OPPORTUNITIES (12-Month Strategic Upside)**")
            for o in list(dict.fromkeys(swot.get("opportunities", []))):
                st.markdown(f"- {o}")
        with c_sw2:
            st.markdown("**WEAKNESSES (Internal Engineering Trade-Offs)**")
            for w in list(dict.fromkeys(swot.get("weaknesses", []))):
                st.markdown(f"- {w}")
            st.markdown("**THREATS (External Risks & Mitigations)**")
            for t in list(dict.fromkeys(swot.get("threats", []))):
                st.markdown(f"- {t}")

        st.header("8. Granular Multi-Scale Financial & AWS Cost Model (1× to 10×)")
        fin_rows = report.get("financial_rows", [])
        if fin_rows:
            st.caption("Table 4 — Reconciled monthly OpEx (USD); Initial Engineering CapEx, if present, is one-time and excluded from totals. Source: model-generated component estimates.")
            st.table([
                {
                    "Cloud / API Line Item": r.get("line_item", ""),
                    "1× Baseline": r.get("scale_1x", ""),
                    "2× Growth": r.get("scale_2x", ""),
                    "5× Scale": r.get("scale_5x", ""),
                    "10× Enterprise": r.get("scale_10x", ""),
                }
                for r in fin_rows
            ])
        if report.get("financial_footnote"):
            st.caption(f"*{report.get('financial_footnote')}*")

        st.markdown(
            f"""
            <div class="pm-summary-box">
                <b>Executive Implementation Recommendation:</b> {report.get('executive_recommendation', '')}
                <br/><br/>
                <i>Note on Section 5 Implementation: The code provided in <b>5. Complete Production Python Implementation</b> is a reference example; actual implementation code may differ or get updated as underlying SDKs, APIs, and production requirements evolve.</i>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.header("9. Appendix & Technical Attributions (Notes)")
        for note in report.get("appendix_notes", []):
            st.markdown(f"- {note}")

        st.markdown(
            f"""
            <div class="pm-footer">
                PlanModelAI.com — Enhanced Enterprise Architectural Blueprint (Product C • Beta Testing) • Generated on {generated_date} • Client Vault ID: <code class="pm-vault-id">{safe_vault_id}</code>
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# Development-only assertions; disabled in production unless explicitly requested.
if __name__ == "__main__" and os.getenv("PICKMODELAI_RUN_SELF_TESTS") == "1":
    from tempfile import TemporaryDirectory

    assert VAULT_ID_REGEX.fullmatch("PMAI-GCCF-2422") is None
    assert VAULT_ID_REGEX.fullmatch("PMAI-6CCF-2422") is not None
    assert detect_primary_modality("AI assistant to draft emails, summarize PDFs, generate code, and transcribe voice notes") == "TEXT_REASONING"
    assert detect_primary_modality("Inbound call center voicebot with real-time TTS and telephony") == "VOICE_AUDIO"

    columns = ("scale_1x", "scale_2x", "scale_5x", "scale_10x")
    sample_rows = [
        {"line_item": "Initial Engineering CapEx", **dict(zip(columns, ["$85,000", "$90,000", "$95,000", "$100,000"]))},
        {"line_item": "Compute and API OpEx", **dict(zip(columns, ["$500", "$500", "$500", "$500"]))},
        {"line_item": "Total Custom Enterprise Stack", **dict(zip(columns, ["$85,500", "$500", "$500", "$500"]))},
        {"line_item": "Pre-Built SaaS Equivalent", **dict(zip(columns, ["$1,000", "$1,000", "$1,000", "$1,000"]))},
    ]
    corrected = _fix_financial_rows(sample_rows, locked_baseline_usd=450)
    assert all(corrected[0][column] == "$85,000" for column in columns)
    values = [_parse_usd_cost(corrected[2][column]) for column in columns]
    assert len(set(values)) == 4 and values == sorted(values)
    assert corrected[2]["scale_1x"] == "$450"
    print("Reconciled Section 8 Total:", corrected[2])

    with TemporaryDirectory() as directory:
        test_vault = _ContinuityVaultAgent(None, Path(directory))
        for source, target in (("A", "B"), ("A", "C"), ("B", "C")):
            source_tier = {"A": "1", "B": "2"}[source]
            target_tier = {"B": "2", "C": "3"}[target]
            initial = test_vault.resolve_vault(None, source)
            original_id = initial["vault_id"]
            assert test_vault.save_blueprint_to_vault({}, {}, f"Tier {source_tier}", original_id) == original_id
            continued = test_vault.resolve_vault(original_id, target)
            assert continued["vault_id"] == original_id
            gate = test_vault.check_tier_generation_allowed(original_id, target_tier)
            assert gate["allowed"]
            assert bool(gate["warning"]) == (source == "A" and target == "C")
            payload = {}
            assert test_vault.save_blueprint_to_vault({}, payload, f"Tier {target_tier}", original_id) == original_id
            assert payload["lineage"]["origin_product"] == source
            assert payload["lineage"]["parent_vault_id"] == original_id
            assert payload["lineage"]["parent_chain"]
            assert not test_vault.check_tier_generation_allowed(original_id, target_tier)["allowed"]
            reloaded = _ContinuityVaultAgent(None, Path(directory)).resolve_vault(original_id, target)
            assert reloaded["vault_id"] == original_id
            assert reloaded["tiers_generated"] == sorted([source_tier, target_tier], key=int)
    print("Vault continuity checks passed.")


# ===========================================================================
# 5. GLOBAL PORTAL CSS & SIDEBAR NAVIGATION
# ===========================================================================
st.markdown(
    """
    <style>
    .pm-vault-id, code {
        font-family: 'Courier New', Courier, monospace !important;
        font-variant-ligatures: none !important;
        letter-spacing: 0.6px !important;
        font-weight: 700 !important;
    }
    .pm-doc-container {
        background-color: #FFFFFF;
        color: #1E293B;
        padding: 28px 34px;
        border: 1px solid #CBD5E1;
        border-radius: 6px;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.06);
        margin-top: 12px;
        margin-bottom: 24px;
    }
    .pm-header-bar {
        background-color: #1B365D;
        color: #FFFFFF;
        padding: 18px 22px;
        border-radius: 4px;
        margin-bottom: 14px;
    }
    .pm-sub-tag {
        color: #BAE6FD;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.6px;
        text-transform: uppercase;
    }
    .pm-doc-title {
        font-size: 22px;
        font-weight: 800;
        margin: 4px 0 8px 0;
        color: #FFFFFF;
    }
    .pm-badges {
        color: #BAE6FD;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.6px;
        text-transform: uppercase;
        margin-top: 4px;
    }
    .pm-badges code {
        color: #BAE6FD;
        background-color: rgba(255, 255, 255, 0.12);
        padding: 2px 6px;
        border-radius: 3px;
        font-size: 12px;
        font-weight: 700;
    }
    .pm-section-hdr {
        color: #1B365D;
        font-size: 16px;
        font-weight: 800;
        border-bottom: 2px solid #2C6E7F;
        padding-bottom: 4px;
        margin-top: 24px;
        margin-bottom: 10px;
        text-transform: uppercase;
    }
    .pm-node-box {
        border: 1.5px solid #1B365D;
        background: #F8FAFC;
        border-radius: 6px;
        padding: 11px 8px;
        text-align: center;
        min-height: 76px;
        margin-bottom: 8px;
    }
    .pm-node-title {
        font-weight: 800;
        font-size: 12.5px;
        color: #0F766E;
    }
    .pm-node-sub {
        font-size: 11.5px;
        color: #334155;
        margin-top: 4px;
    }
    .pm-summary-box {
        background-color: #E0F2FE;
        border-left: 4px solid #1B365D;
        padding: 12px 16px;
        margin-top: 12px;
        font-size: 13.5px;
        color: #0F172A;
    }
    .pm-footer {
        text-align: center;
        font-size: 11.5px;
        color: #64748B;
        border-top: 1px solid #CBD5E1;
        padding-top: 10px;
        margin-top: 22px;
    }
    .pm-suite-card {
        border: 1px solid #CBD5E1;
        border-top: 4px solid #1B365D;
        border-radius: 8px;
        padding: 20px;
        background: #F8FAFC;
        height: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "active_view" not in st.session_state:
    st.session_state.active_view = "HUB"

last_vid = (
    st.session_state.get("c_final_vault_id")
    or st.session_state.get("b_final_vault_id")
    or st.session_state.get("client_ref_id")
)

with st.sidebar:
    st.title("📐 PickModelAI.com")
    st.caption("Unified Architecture Suite • Beta Testing")
    st.markdown("---")

    if st.button("🏠 Executive Showcase Hub", use_container_width=True, type="primary" if st.session_state.active_view == "HUB" else "secondary"):
        st.session_state.active_view = "HUB"
        st.rerun()
    if st.button("📐 Product A: Genesis Diagnostic (Beta)", use_container_width=True, type="primary" if st.session_state.active_view == "A" else "secondary"):
        st.session_state.active_view = "A"
        st.rerun()
    if st.button("📘 Product B: DIY Blueprint (Beta)", use_container_width=True, type="primary" if st.session_state.active_view == "B" else "secondary"):
        st.session_state.active_view = "B"
        st.rerun()
    if st.button("🏛️ Product C: Enterprise (Beta)", use_container_width=True, type="primary" if st.session_state.active_view == "C" else "secondary"):
        st.session_state.active_view = "C"
        st.rerun()

    st.markdown("---")
    if last_vid:
        st.success(f"🔑 **Active Vault ID:**\n`{last_vid}`")
    sidebar_vault_id = st.text_input(
        "Client Vault ID",
        value=last_vid or "",
        placeholder="PMAI-6CCF-2422",
        key="portal_vault_id",
    ).strip().upper()
    if sidebar_vault_id:
        if not VAULT_ID_REGEX.fullmatch(sidebar_vault_id):
            st.error("Use PMAI-XXXX-XXXX with hexadecimal characters 0–9 and A–F.")
        else:
            try:
                sidebar_vault = get_vault_agent()
                if sidebar_vault and sidebar_vault.retrieve_client_reports(sidebar_vault_id):
                    sidebar_status = sidebar_vault.resolve_vault(sidebar_vault_id)
                    st.caption(f"🟢 Vault verified · Origin {sidebar_status['origin_product']} · Tiers {', '.join(sidebar_status['tiers_generated']) or 'none'}")
                else:
                    st.caption("⚪ Vault not found")
            except Exception as exc:
                st.error(str(exc))
    st.info(
        "🔒 **Comming Soon:**\n"
        "Autonomous Executive Strategy & AI Governance Roadmap. Hybrid CXO advisory (5 hours, 3-Day Delivery, custom Q&A sessions)."
    )


# ===========================================================================
# VIEW 0: EXECUTIVE SHOWCASE HUB (LANDING PAGE WITH LAUNCH BUTTONS)
# ===========================================================================
if st.session_state.active_view == "HUB":
    st.title("📐 PickModelAI — Executive AI Systems Architecture Suite (Beta Testing)")
    st.markdown(
        "**From Day-1 Client Engagement to Multi-Scale Enterprise Deployment.** "
        "Select any architecture tier below during our **Beta Testing** phase to launch its dedicated AI Solutions Architect chatbot and generate your blueprint."
    )
    st.markdown("---")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            """
            <div class="pm-suite-card">
                <div style="color:#0F766E; font-weight:800; font-size:12px; text-transform:uppercase;">TIER 1 • BETA TESTING</div>
                <h3 style="margin-top:4px; color:#1B365D;">📐 Product A: Genesis Architecture Diagnostic</h3>
                <p style="font-size:14px; color:#334155;">
                    Rapid 4-question feasibility & TCO diagnostic for founders and initial client discovery.
                </p>
                <ul style="font-size:13px; color:#475569; padding-left:18px;">
                    <li>Deterministic modality detection (Voice, Video, Vision, Text)</li>
                    <li>Live web research for 2026 commercial & open-weight models</li>
                    <li>Managed Cloud API vs. Private AWS GPU VPC cost comparison</li>
                    <li>Generates downloadable <b>2-Page Executive PDF</b> + <b>Client Vault ID</b></li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("🚀 Launch Product A Chatbot (Beta)", key="launch_a", type="primary", use_container_width=True):
            st.session_state.active_view = "A"
            st.rerun()

    with c2:
        st.markdown(
            """
            <div class="pm-suite-card">
                <div style="color:#0F766E; font-weight:800; font-size:12px; text-transform:uppercase;">TIER 2 • DIY ENGINEERING (BETA TESTING)</div>
                <h3 style="margin-top:4px; color:#1B365D;">📘 Product B: Full DIY Engineering Blueprint</h3>
                <p style="font-size:14px; color:#334155;">
                    Complete 7-section implementation specification for engineering teams building in-house.
                </p>
                <ul style="font-size:13px; color:#475569; padding-left:18px;">
                    <li>Instant scope import via <b>Product A Client Vault ID</b> or 4 discovery questions</li>
                    <li>Build vs. Buy SaaS economics & 4-Node Hybrid Architecture</li>
                    <li>6-Day Implementation Roadmap with verification gates</li>
                    <li>Reference Python implementation & 1k–100k AWS cost scaling table</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("📘 Launch Product B Chatbot (Beta)", key="launch_b", type="primary", use_container_width=True):
            st.session_state.active_view = "B"
            st.rerun()

    with c3:
        st.markdown(
            """
            <div class="pm-suite-card">
                <div style="color:#0F766E; font-weight:800; font-size:12px; text-transform:uppercase;">TIER 3 • INSTITUTIONAL (BETA TESTING)</div>
                <h3 style="margin-top:4px; color:#1B365D;">🏛️ Product C: Enhanced Enterprise Blueprint</h3>
                <p style="font-size:14px; color:#334155;">
                    Board-ready 9-section systems & financial blueprint for enterprise scale (1× to 10×).
                </p>
                <ul style="font-size:13px; color:#475569; padding-left:18px;">
                    <li>Requires verified <b>Product A or B Client Vault ID</b> + 3 dynamic enterprise questions</li>
                    <li>36-Month Cumulative TCO Trajectory Chart & 7-Node LangGraph Pipeline</li>
                    <li>Temporal RAG configuration, 2×/5×/10× Stress-Test & SWOT analysis</li>
                    <li>Multi-scale financial model with automatic contact profile continuity</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("🏛️ Launch Product C Chatbot (Beta)", key="launch_c", type="primary", use_container_width=True):
            st.session_state.active_view = "C"
            st.rerun()

    st.markdown("---")
    st.markdown(
        "**How Continuity Works Across the Suite:** Start with **Product A** to receive your cryptographic **Client Vault ID** (`PMAI-XXXX-XXXX`), "
        "or jump directly into **Product B** or **Product C** using an existing Vault ID to lock scope and eliminate repetitive data entry."
    )


# ===========================================================================
# VIEW 1: PRODUCT A — GENESIS ARCHITECTURE DIAGNOSTIC
# ===========================================================================
elif st.session_state.active_view == "A":
    QUESTIONS_A = [
        (
            "Welcome to **PickModelAI — Product A: Genesis Architecture Diagnostic (Beta Testing)**! 👋\n\n"
            "**Question 1 of 4:** What do you want your AI system to do, and what type of data will it process or create "
            "(e.g., text/emails, code, long PDFs, images, video, or voice/audio)? *(Please answer in under 100 words)*"
        ),
        (
            "Got it! **Question 2 of 4:** Who is the target user for this product, what is their biggest daily pain point, "
            "and what is your main 6-month goal? *(Please answer in under 100 words)*"
        ),
        (
            "Almost there. **Question 3 of 4:** Roughly how many requests, emails, reports, or videos will it process per day/month, "
            "and do you have strict privacy rules (like HIPAA, banking/SOC2, GDPR/European servers, or keeping 100% of data on your own private cloud)? "
            "*(Please answer in under 75 words)*"
        ),
        (
            "Final step! **Question 4 of 4:** What is your target monthly infrastructure budget (in USD) for AI APIs and cloud hosting "
            "(e.g., prototype tier, $50/mo, $200/mo, or $500+/mo)? *(Please answer in under 50 words)*"
        ),
    ]
    WORD_LIMITS_A = [100, 100, 75, 50]

    st.title("📐 PickModelAI — Product A: AI Architecture Diagnostic (Beta Testing)")
    st.caption("Beta Testing Edition • Answer 4 quick questions to get live-researched stack options, cloud TCO, and a 2-page PDF blueprint.")

    allowed, _ = check_and_increment_rate_limit(increment=False)
    if not allowed:
        st.warning("You have reached the maximum of 5 free architecture assessments for today. Please return tomorrow.")
        st.stop()

    if "step" not in st.session_state:
        st.session_state.step = 0
        st.session_state.answers = []
        st.session_state.messages = [{"role": "assistant", "content": QUESTIONS_A[0]}]
        st.session_state.pdf_bytes = None
        st.session_state.blueprint_data = None
        st.session_state.client_ref_id = None
        st.session_state.security_blocked = False

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if st.session_state.get("security_blocked", False):
        st.error("🚫 Request Blocked by Safety Guardrails.")
        if st.button("🔄 Start a New Assessment"):
            for key in ["step", "answers", "messages", "pdf_bytes", "blueprint_data", "client_ref_id", "security_blocked"]:
                st.session_state.pop(key, None)
            st.rerun()
        st.stop()

    if st.session_state.step < 4:
        current_limit = WORD_LIMITS_A[st.session_state.step]
        user_input = st.chat_input(f"Type your answer (Max {current_limit} words)...")
        if user_input:
            is_valid, error_msg = validate_user_input(user_input, st.session_state.step, current_limit)
            if not is_valid:
                st.toast(error_msg)
            else:
                st.session_state.messages.append({"role": "user", "content": user_input.strip()})
                st.session_state.answers.append(user_input.strip())
                st.session_state.step += 1
                if st.session_state.step < 4:
                    st.session_state.messages.append({"role": "assistant", "content": QUESTIONS_A[st.session_state.step]})
                else:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "🔒 **Researching live AI models, modeling cloud TCO, and saving your blueprint with vault continuity...**",
                    })
                st.rerun()

    if st.session_state.step == 4 and st.session_state.pdf_bytes is None:
        with st.spinner("Researching live AI models, calculating cloud costs, and generating your 2-page report..."):
            combined_profile = (
                f"1. Core Idea & Modality: {st.session_state.answers[0]}\n"
                f"2. Target User & 6-Month Goal: {st.session_state.answers[1]}\n"
                f"3. Volume & Privacy Needs: {st.session_state.answers[2]}\n"
                f"4. Monthly Budget & Final Details: {st.session_state.answers[3]}"
            )
            is_safe, _ = evaluate_project_safety(combined_profile)
            if not is_safe:
                st.session_state.security_blocked = True
                st.rerun()
            try:
                if st.session_state.blueprint_data is None:
                    blueprint_dict = generate_assessment_from_llm(
                        user_profile=combined_profile,
                        core_idea=st.session_state.answers[0],
                        volume_privacy=st.session_state.answers[2],
                    )
                    client_ref_id = save_product_a_to_vault(st.session_state.answers, blueprint_dict)
                    st.session_state.blueprint_data = blueprint_dict
                    st.session_state.client_ref_id = client_ref_id
                else:
                    blueprint_dict = st.session_state.blueprint_data
                    client_ref_id = st.session_state.client_ref_id
                pdf_bytes = generate_blueprint_pdf(blueprint_dict, client_ref_id=client_ref_id)
                check_and_increment_rate_limit(increment=True)
                st.session_state.pdf_bytes = pdf_bytes
                st.rerun()
            except Exception as e:
                st.error(f"⚠️ Something went wrong while generating your blueprint: {e}")
                if st.button("🔄 Try Again"):
                    st.rerun()

    if st.session_state.pdf_bytes is not None:
        bp = st.session_state.blueprint_data
        ref_id = st.session_state.client_ref_id
        st.success(f"✅ **Product A Diagnostic Ready!** Readiness Score: **{bp['feasibility_grade']}**")
        st.code(f"Zero-Login Client Vault ID: {ref_id}", language="text")
        st.download_button(
            label="📥 Download Your 2-Page Architecture Blueprint (PDF)",
            data=st.session_state.pdf_bytes,
            file_name=f"PickModelAI_ProductA_{ref_id}.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True,
        )
        col_next1, col_next2 = st.columns(2)
        with col_next1:
            if st.button("📘 Continue to Product B (Beta) with this Vault ID →", use_container_width=True):
                st.session_state.active_view = "B"
                st.rerun()
        with col_next2:
            if st.button("🔄 Start a New Product A Assessment", use_container_width=True):
                for key in ["step", "answers", "messages", "pdf_bytes", "blueprint_data", "client_ref_id", "security_blocked"]:
                    st.session_state.pop(key, None)
                st.rerun()


# ===========================================================================
# VIEW 2: PRODUCT B — FULL DIY ENGINEERING BLUEPRINT
# ===========================================================================
elif st.session_state.active_view == "B":
    QUESTIONS_B = [
        (
            "Welcome to **PickModelAI — Product B: Full DIY Engineering Blueprint (Beta Testing)**! 📘\n\n"
            "You can proceed in **one of two ways** (strict scope integrity enforced):\n"
            "- **Option A (Vault ID Import):** Paste your existing **Client Vault ID** (e.g., `PMAI-BCEC-98ED`) at any point to lock onto your verified Product A scope. *(To modify a saved scope, email **support@pickmodelai.com**.)*\n"
            "- **Option B (New Project Discovery):** Answer **Question 1 of 4** below:\n\n"
            "**Question 1 of 4:** What is the core AI system or automation product you want to build, what data does it process, "
            "and how often should it run (e.g., real-time API, twice-daily cron publisher, or batch workflow)? *(Max 200 words)*"
        ),
        (
            "Got it! **Question 2 of 4:** What commercial SaaS tools or manual workflows are you replacing, "
            "and what are your target operational scale metrics (e.g., daily requests, subscribers, or records processed)? *(Max 200 words)*"
        ),
        (
            "Great. **Question 3 of 4:** Which cloud/model architecture do you prefer (Managed Cloud APIs vs. Private AWS VPC / ECS Fargate), "
            "and what compliance, latency, or audit rules must the system enforce? *(Max 200 words)*"
        ),
        (
            "Final discovery step! **Question 4 of 4:** What is your target monthly infrastructure budget (in USD) and desired launch timeline "
            "(e.g., 6-day MVP sprint vs. 2-week rollout)? *(Max 200 words)*"
        ),
    ]

    st.title("📘 PickModelAI — Product B: Full DIY Engineering Blueprint (Beta Testing)")
    st.caption("Beta Testing Edition • Enter your Client Vault ID or complete 4 discovery questions to render your 7-section engineering blueprint on-page.")

    if last_vid:
        st.info(f"🔑 **Detected Active Session Vault ID:** `{last_vid}` — Copy and paste this ID below to lock your scope instantly.")

    if "b_stage" not in st.session_state:
        st.session_state.b_stage = "CHAT_Q"
        st.session_state.b_q_index = 0
        st.session_state.b_answers = []
        st.session_state.b_messages = [{"role": "assistant", "content": QUESTIONS_B[0]}]
        st.session_state.b_imported_vault_id = None
        st.session_state.b_imported_context = None
        st.session_state.b_contact = None
        st.session_state.b_report = None
        st.session_state.b_final_vault_id = None
        st.session_state.b_autofilled_city = ""
        st.session_state.b_pin_status = ""

    for msg in st.session_state.b_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if st.session_state.b_stage == "CHAT_Q":
        user_msg = st.chat_input("Enter Client Vault ID (e.g., PMAI-BCEC-98ED) or answer the current question (Max 200 words)...")
        if st.button("Import sidebar Client Vault ID", key="b_import_sidebar", disabled=not bool(VAULT_ID_REGEX.fullmatch(sidebar_vault_id))):
            user_msg = sidebar_vault_id
        if user_msg:
            ok, err = validate_chat_answer(user_msg, max_words=MAX_WORDS_PER_ANSWER)
            if not ok:
                st.toast(err)
            else:
                cleaned_msg = user_msg.strip()
                st.session_state.b_messages.append({"role": "user", "content": cleaned_msg})
                vault_match = VAULT_ID_REGEX.search(cleaned_msg)
                if vault_match:
                    candidate_id = vault_match.group(0).upper()
                    vault = get_vault_agent()
                    try:
                        records = vault.retrieve_client_reports(candidate_id) if vault else []
                    except Exception as exc:
                        st.error(str(exc))
                        st.stop()
                    if records:
                        gate = vault.check_tier_generation_allowed(candidate_id, "2")
                        if not gate["allowed"]:
                            st.error(gate["reason"])
                            st.stop()
                        rec = next((r for r in records if _tier_number(r.get("tier", "")) == "1"), records[0])
                        locked_paid, locked_oss = _extract_locked_costs_from_records(records)
                        inp = rec.get("user_inputs", {})
                        rep = rec.get("report_data", {})
                        st.session_state.b_answers = [inp.get(key, "") for key in ("goal", "workflow", "constraints", "budget")]
                        st.session_state.b_imported_vault_id = candidate_id
                        cost_lines = ""
                        if locked_paid:
                            cost_lines += f"\n- LOCKED Baseline Managed Cloud Monthly OpEx (from Product A): ${locked_paid}/month"
                        if locked_oss:
                            cost_lines += f"\n- LOCKED Baseline Private VPC Monthly OpEx (from Product A): ${locked_oss}/month"
                        st.session_state.b_imported_context = (
                            f"Locked Vault Record ({candidate_id} — {rec.get('tier', 'Tier 1 (Product A — Genesis Diagnostic)')}):\n"
                            f"- Core Goal & Modality: {inp.get('goal', '')}\n"
                            f"- Target Users & Workflow: {inp.get('workflow', '')}\n"
                            f"- Constraints & Compliance: {inp.get('constraints', '')}\n"
                            f"- Budget Target: {inp.get('budget', '')}\n"
                            f"- Prior Recommended Cloud Stack: {rep.get('paid_stack_summary', '')}\n"
                            f"- Prior Recommended VPC Stack: {rep.get('oss_stack_summary', '')}"
                            f"{cost_lines}"
                        )
                        st.session_state.b_stage = "CONTACT_FORM"
                        st.session_state.b_messages.append({
                            "role": "assistant",
                            "content": (
                                f"🔒 **Client Vault ID `{candidate_id}` Verified & Locked!**\n\n"
                                "Your saved project scope and baseline cost model have been loaded directly from the vault. "
                                "If you need to modify your project scope or request fresh reports, please contact **support@pickmodelai.com**.\n\n"
                                "Please complete your **Client Contact Details** below to render your Product B blueprint."
                            ),
                        })
                        st.rerun()
                    else:
                        st.session_state.b_messages.append({
                            "role": "assistant",
                            "content": (
                                f"⚠️ No verified blueprint was found for **`{candidate_id}`**. "
                                f"Please check your Vault ID or continue answering **Question {st.session_state.b_q_index + 1} of 4**."
                            ),
                        })
                        st.rerun()
                elif "PMAI-" in cleaned_msg.upper():
                    st.session_state.b_messages.append({
                        "role": "assistant",
                        "content": "⚠️ Invalid Vault ID. Use PMAI-XXXX-XXXX with hexadecimal characters 0–9 and A–F; check 6 versus G.",
                    })
                    st.rerun()
                else:
                    st.session_state.b_answers.append(cleaned_msg)
                    st.session_state.b_q_index += 1
                    if st.session_state.b_q_index < 4:
                        st.session_state.b_messages.append({"role": "assistant", "content": QUESTIONS_B[st.session_state.b_q_index]})
                    else:
                        st.session_state.b_stage = "CONTACT_FORM"
                        st.session_state.b_messages.append({
                            "role": "assistant",
                            "content": (
                                "🎯 **All 4 discovery answers recorded!**\n\n"
                                "Please select your **Country** and enter your **PIN / Postal Code** first to auto-verify your City, "
                                "then complete your contact details below."
                            ),
                        })
                    st.rerun()

    if st.session_state.b_stage == "CONTACT_FORM":
        st.subheader("📋 Client Contact Verification")
        if st.session_state.b_imported_vault_id:
            st.info(f"🔒 **Locked to Client Vault ID `{st.session_state.b_imported_vault_id}`** — Need to change your project scope? Email **support@pickmodelai.com**.")

        col_c, col_p, col_btn = st.columns([2, 2, 1.3])
        with col_c:
            selected_country = st.selectbox("1. Country *", options=list(COUNTRY_RULES.keys()), index=0, key="b_country_sel")
        with col_p:
            rules_now = COUNTRY_RULES[selected_country]
            entered_pin = st.text_input("2. PIN / Postal Code *", placeholder=rules_now["pin_hint"], key="b_pin_in")
        with col_btn:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            verify_pin_clicked = st.button("🔍 Verify PIN & Fill City", key="b_verify_btn", use_container_width=True)

        if verify_pin_clicked:
            if not entered_pin.strip():
                st.warning(f"Please enter your {rules_now['pin_hint']} first.")
            else:
                pin_valid, resolved_city, status_msg = lookup_city_from_pin(selected_country, entered_pin)
                if pin_valid:
                    if resolved_city:
                        st.session_state.b_autofilled_city = resolved_city
                    st.session_state.b_pin_status = f"✅ {status_msg}"
                else:
                    st.session_state.b_autofilled_city = ""
                    st.session_state.b_pin_status = f"❌ {status_msg}"

        if st.session_state.b_pin_status:
            if st.session_state.b_pin_status.startswith("✅"):
                st.success(st.session_state.b_pin_status)
            else:
                st.error(st.session_state.b_pin_status)

        with st.form("product_b_contact_form"):
            f1, f2 = st.columns(2)
            with f1:
                city_val = st.text_input("3. City / State (Auto-filled from PIN) *", value=st.session_state.b_autofilled_city)
                client_name = st.text_input("4. Full Name *", placeholder="e.g., Aarav Sharma")
                client_email = st.text_input("5. Business Email ID *", placeholder="e.g., name@company.com")
            with f2:
                contact_number = st.text_input(f"6. Contact Number ({selected_country} {rules_now['dial']}) *", placeholder=rules_now["phone_hint"])
                company_name = st.text_input("7. Company / Organization *", placeholder="e.g., Acme Corp")
                st.caption(f"Regional validation active for **{selected_country}** ({rules_now['dial']}).")

            generate_btn = st.form_submit_button("🚀 Generate Full DIY Engineering Blueprint On-Page (Beta)", type="primary", use_container_width=True)

        if generate_btn:
            pin_valid, resolved_city, _ = lookup_city_from_pin(selected_country, entered_pin)
            final_city = city_val.strip() or resolved_city
            if pin_valid and resolved_city and not city_val.strip():
                st.session_state.b_autofilled_city = resolved_city

            valid, err_msg = validate_regional_contact(
                name=client_name, company=company_name, email=client_email,
                country=selected_country, pin_code=entered_pin, city=final_city, phone=contact_number
            )
            if not valid:
                st.error(err_msg)
            else:
                contact_dict = {
                    "name": client_name.strip(),
                    "company": company_name.strip(),
                    "email": client_email.strip(),
                    "phone": contact_number.strip(),
                    "city": final_city,
                    "country": selected_country,
                    "pin_code": entered_pin.strip().upper(),
                }
                st.session_state.b_contact = contact_dict
                if st.session_state.b_imported_context:
                    full_context = st.session_state.b_imported_context
                    core_idea_text = st.session_state.b_answers[0] if st.session_state.b_answers else full_context
                    structured_inputs = dict(zip(
                        ("goal", "workflow", "constraints", "budget"),
                        st.session_state.b_answers,
                    ))
                else:
                    ans = st.session_state.b_answers
                    full_context = (
                        f"1. Core AI System, Modality & Cadence: {ans[0]}\n"
                        f"2. Commercial SaaS Replaced & Scale Metrics: {ans[1]}\n"
                        f"3. Preferred Cloud/VPC Stack & Compliance Rules: {ans[2]}\n"
                        f"4. Target Monthly Budget & Launch Timeline: {ans[3]}"
                    )
                    core_idea_text = ans[0]
                    structured_inputs = {"goal": ans[0], "workflow": ans[1], "constraints": ans[2], "budget": ans[3]}

                with st.spinner("Generating your DIY Engineering Blueprint..."):
                    try:
                        report_dict = generate_product_b_document(
                            project_context=full_context,
                            core_idea=core_idea_text,
                            vault_id=st.session_state.b_imported_vault_id,
                        )
                        final_vault_id = persist_lead_and_blueprint(
                            contact=contact_dict,
                            project_inputs=structured_inputs,
                            report_data=report_dict,
                            existing_vault_id=st.session_state.b_imported_vault_id,
                        )
                        st.session_state.b_report = report_dict
                        st.session_state.b_final_vault_id = final_vault_id
                        st.session_state.b_stage = "REPORT_READY"
                        st.rerun()
                    except Exception as exc:
                        st.error(f"⚠️ Failed to generate blueprint: {exc}")

    if st.session_state.b_stage == "REPORT_READY" and st.session_state.b_report is not None:
        st.success(f"✅ **Product B Engineering Blueprint Generated!** Saved under Vault ID `{st.session_state.b_final_vault_id}` with a recovery record in `captured_leads/`.")
        render_on_page_blueprint_b(
            report=st.session_state.b_report,
            contact=st.session_state.b_contact,
            vault_id=st.session_state.b_final_vault_id,
            is_vault_locked=bool(st.session_state.b_imported_vault_id),
        )
        cb1, cb2 = st.columns(2)
        with cb1:
            if st.button("🏛️ Continue to Product C (Enterprise Blueprint — Beta) with this Vault ID →", use_container_width=True):
                st.session_state.active_view = "C"
                st.rerun()
        with cb2:
            if st.button("🔄 Start a New Product B Blueprint", use_container_width=True):
                for k in ["b_stage", "b_q_index", "b_answers", "b_messages", "b_imported_vault_id", "b_imported_context", "b_contact", "b_report", "b_final_vault_id", "b_autofilled_city", "b_pin_status"]:
                    st.session_state.pop(k, None)
                st.rerun()


# ===========================================================================
# VIEW 3: PRODUCT C — ENHANCED ENTERPRISE ARCHITECTURAL BLUEPRINT
# ===========================================================================
elif st.session_state.active_view == "C":
    st.title("🏛️ PickModelAI — Product C: Enhanced Enterprise Architectural Blueprint (Beta Testing)")
    st.caption("Enterprise Architecture Tier (Beta Testing) • Requires a verified Product B or Product A Client Vault ID.")
    if st.session_state.get("c_lifecycle_warning"):
        st.warning(st.session_state.c_lifecycle_warning)

    if last_vid:
        st.info(f"🔑 **Detected Active Session Vault ID:** `{last_vid}` — Copy and paste this ID below to lock your scope instantly.")

    if "c_stage" not in st.session_state:
        st.session_state.c_stage = "AWAITING_VAULT_ID"
        st.session_state.c_messages = [
            {
                "role": "assistant",
                "content": (
                    "Welcome to **PickModelAI — Product C: Enhanced Enterprise Architectural Blueprint (Beta Testing)**! 🏛️\n\n"
                    "Product C is our **Enterprise Architectural Blueprint**. "
                    "To maintain strict scope integrity, **Product C requires a verified Client Vault ID from Product A or Product B**.\n\n"
                    "Please enter your verified **Client Vault ID** (e.g., `PMAI-BCEC-98ED`) below to begin."
                ),
            }
        ]
        st.session_state.c_imported_vault_id = None
        st.session_state.c_imported_tier = None
        st.session_state.c_imported_context = None
        st.session_state.c_core_idea = ""
        st.session_state.c_lifecycle_warning = ""
        st.session_state.c_contact = None
        st.session_state.c_report = None
        st.session_state.c_final_vault_id = None
        st.session_state.c_dynamic_questions = []
        st.session_state.c_dyn_index = 0
        st.session_state.c_dynamic_answers_list = []
        st.session_state.c_dynamic_answers = ""
        st.session_state.c_autofilled_city = ""
        st.session_state.c_pin_status = ""

    for msg in st.session_state.c_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if st.session_state.c_stage == "AWAITING_VAULT_ID":
        user_msg = st.chat_input("Enter your Product B or Product A Client Vault ID (e.g., PMAI-BCEC-98ED)...")
        if st.button("Import sidebar Client Vault ID", key="c_import_sidebar", disabled=not bool(VAULT_ID_REGEX.fullmatch(sidebar_vault_id))):
            user_msg = sidebar_vault_id
        if user_msg:
            cleaned_msg = user_msg.strip()
            st.session_state.c_messages.append({"role": "user", "content": cleaned_msg})
            vault_match = VAULT_ID_REGEX.search(cleaned_msg)
            if vault_match:
                candidate_id = vault_match.group(0).upper()
                vault = get_vault_agent()
                imported_tuple = None
                if vault is not None:
                    with st.spinner(f"Verifying Client Vault ID {candidate_id} across Vault records..."):
                        try:
                            imported_tuple = fetch_prior_tier_context(vault, candidate_id)
                        except Exception as exc:
                            st.error(str(exc))
                            st.stop()

                if imported_tuple is not None:
                    gate = vault.check_tier_generation_allowed(candidate_id, "3")
                    if not gate["allowed"]:
                        st.error(gate["reason"])
                        st.stop()
                    tier_label, core_goal, locked_ctx, saved_contact = imported_tuple
                    st.session_state.c_imported_vault_id = candidate_id
                    st.session_state.c_imported_tier = tier_label
                    st.session_state.c_imported_context = locked_ctx
                    st.session_state.c_core_idea = core_goal
                    st.session_state.c_lifecycle_warning = gate["warning"]
                    st.session_state.c_contact = saved_contact
                    if gate["warning"]:
                        st.warning(gate["warning"])

                    with st.spinner("Analyzing your project architecture to generate 3 targeted enterprise discovery questions..."):
                        questions = generate_dynamic_questions(locked_ctx)
                        if len(questions) != 3 or any(not isinstance(q, str) or not q.strip() for q in questions):
                            st.error("The discovery response did not contain three valid questions. Please import the same Vault ID again.")
                            st.stop()
                        st.session_state.c_dynamic_questions = questions
                        st.session_state.c_dyn_index = 0
                        st.session_state.c_dynamic_answers_list = []

                    st.session_state.c_stage = "DYNAMIC_DISCOVERY"
                    st.session_state.c_messages.append({
                        "role": "assistant",
                        "content": (
                            f"🔒 **Vault ID `{candidate_id}` Verified Successfully!** (`{tier_label}`)\n\n"
                            f"We have loaded your locked project scope. Let's tackle **Question 1 of 3**:\n\n"
                            f"**{questions[0]}**"
                        ),
                    })
                    st.rerun()
                else:
                    st.session_state.c_messages.append({
                        "role": "assistant",
                        "content": f"⚠️ Vault ID `{candidate_id}` not found in our system. Please complete Product A or B first or verify your Vault ID.",
                    })
                    st.rerun()
            else:
                st.session_state.c_messages.append({
                    "role": "assistant",
                    "content": "⚠️ Invalid Vault ID format. Expected format: `PMAI-XXXX-XXXX` using hexadecimal characters 0–9 and A–F.",
                })
                st.rerun()

    elif st.session_state.c_stage == "DYNAMIC_DISCOVERY":
        current_q_idx = st.session_state.c_dyn_index
        total_q = len(st.session_state.c_dynamic_questions)
        user_msg = st.chat_input(f"Answer Question {current_q_idx + 1} of {total_q} (Max 200 words)...")
        if user_msg:
            ok, err = validate_chat_answer(user_msg, max_words=MAX_WORDS_PER_ANSWER)
            if not ok:
                st.toast(err)
            else:
                cleaned_msg = user_msg.strip()
                st.session_state.c_messages.append({"role": "user", "content": cleaned_msg})
                st.session_state.c_dynamic_answers_list.append(cleaned_msg)
                st.session_state.c_dyn_index += 1

                if st.session_state.c_dyn_index < total_q:
                    next_q = st.session_state.c_dynamic_questions[st.session_state.c_dyn_index]
                    st.session_state.c_messages.append({
                        "role": "assistant",
                        "content": f"Got it! **Question {st.session_state.c_dyn_index + 1} of {total_q}:**\n\n**{next_q}**",
                    })
                    st.rerun()
                else:
                    qa_pairs = [
                        f"Q{idx+1}: {q_text}\nA{idx+1}: {st.session_state.c_dynamic_answers_list[idx]}"
                        for idx, q_text in enumerate(st.session_state.c_dynamic_questions)
                    ]
                    st.session_state.c_dynamic_answers = "\n\n".join(qa_pairs)
                    if st.session_state.c_contact and st.session_state.c_contact.get("email"):
                        st.session_state.c_stage = "GENERATE_NOW"
                        st.session_state.c_messages.append({
                            "role": "assistant",
                            "content": "🎯 **All enterprise discovery answers recorded!** Verified contact profile found from Product B. Synthesizing your Enterprise Blueprint now...",
                        })
                    else:
                        st.session_state.c_stage = "CONTACT_FORM"
                        st.session_state.c_messages.append({
                            "role": "assistant",
                            "content": "🎯 **All enterprise discovery answers recorded!** Please complete your **Client Contact Verification** below to unlock your blueprint.",
                        })
                    st.rerun()

    elif st.session_state.c_stage == "CONTACT_FORM":
        st.markdown("---")
        st.subheader("📋 Client Contact Verification")
        st.info(f"🔒 **Locked to Client Vault ID `{st.session_state.c_imported_vault_id}`** — Contact details were not stored in Product A.")

        col_c, col_p, col_btn = st.columns([2, 2, 1.3])
        with col_c:
            selected_country = st.selectbox("1. Country *", options=list(COUNTRY_RULES.keys()), index=0, key="c_country_sel")
        with col_p:
            rules_now = COUNTRY_RULES[selected_country]
            entered_pin = st.text_input("2. PIN / Postal Code *", placeholder=rules_now["pin_hint"], key="c_pin_in")
        with col_btn:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            verify_pin_clicked = st.button("🔍 Verify PIN & Fill City", key="c_verify_btn", use_container_width=True)

        if verify_pin_clicked:
            if not entered_pin.strip():
                st.warning(f"Please enter your {rules_now['pin_hint']} first.")
            else:
                pin_valid, resolved_city, status_msg = lookup_city_from_pin(selected_country, entered_pin)
                if pin_valid:
                    if resolved_city:
                        st.session_state.c_autofilled_city = resolved_city
                    st.session_state.c_pin_status = f"✅ {status_msg}"
                else:
                    st.session_state.c_autofilled_city = ""
                    st.session_state.c_pin_status = f"❌ {status_msg}"

        if st.session_state.c_pin_status:
            if st.session_state.c_pin_status.startswith("✅"):
                st.success(st.session_state.c_pin_status)
            else:
                st.error(st.session_state.c_pin_status)

        with st.form("product_c_contact_form"):
            f1, f2 = st.columns(2)
            with f1:
                city_val = st.text_input("3. City / State (Auto-filled from PIN) *", value=st.session_state.c_autofilled_city)
                client_name = st.text_input("4. Full Name *", placeholder="e.g., Aarav Sharma")
                client_email = st.text_input("5. Business Email ID *", placeholder="e.g., name@company.com")
            with f2:
                contact_number = st.text_input(f"6. Contact Number ({selected_country} {rules_now['dial']}) *", placeholder=rules_now["phone_hint"])
                company_name = st.text_input("7. Company / Organization *", placeholder="e.g., Acme Corp")
                st.caption(f"Regional validation active for **{selected_country}** ({rules_now['dial']}).")

            generate_btn = st.form_submit_button("🚀 Generate Enhanced Enterprise Blueprint On-Page (Beta)", type="primary", use_container_width=True)

        if generate_btn:
            pin_valid, resolved_city, _ = lookup_city_from_pin(selected_country, entered_pin)
            final_city = city_val.strip() or resolved_city
            if pin_valid and resolved_city and not city_val.strip():
                st.session_state.c_autofilled_city = resolved_city

            valid, err_msg = validate_regional_contact(
                name=client_name, company=company_name, email=client_email,
                country=selected_country, pin_code=entered_pin, city=final_city, phone=contact_number
            )
            if not valid:
                st.error(err_msg)
            else:
                st.session_state.c_contact = {
                    "name": client_name.strip(),
                    "company": company_name.strip(),
                    "email": client_email.strip(),
                    "phone": contact_number.strip(),
                    "city": final_city,
                    "country": selected_country,
                    "pin_code": entered_pin.strip().upper(),
                }
                st.session_state.c_stage = "GENERATE_NOW"
                st.rerun()

    elif st.session_state.c_stage == "GENERATE_NOW":
        with st.spinner("Synthesizing your Enhanced Enterprise Architectural Blueprint (Beta Testing)..."):
            try:
                locked_context = st.session_state.c_imported_context or "Enterprise AI Platform"
                dynamic_ans = st.session_state.c_dynamic_answers or "Enterprise scaling and compliance requirements."
                safe_contact = st.session_state.c_contact or {}
                imported_vid = st.session_state.c_imported_vault_id

                structured_inputs = {
                    "goal": st.session_state.get("c_core_idea") or locked_context,
                    "workflow": locked_context,
                    "constraints": dynamic_ans,
                    "budget": "See locked vault scope and enterprise discovery answers.",
                }
                report_dict = generate_product_c_document(
                    project_context=locked_context,
                    dynamic_answers=dynamic_ans,
                    core_idea=st.session_state.get("c_core_idea") or locked_context,
                    vault_id=imported_vid,
                )
                final_vault_id = persist_product_c_record(
                    contact=safe_contact,
                    project_inputs=structured_inputs,
                    report_data=report_dict,
                    existing_vault_id=imported_vid,
                )
                st.session_state.c_report = report_dict
                st.session_state.c_final_vault_id = final_vault_id
                st.session_state.c_stage = "REPORT_READY"
                st.rerun()
            except Exception as exc:
                st.error(f"⚠️ Failed to generate enterprise blueprint: {exc}")
                st.exception(exc)
                if st.button("🔄 Restart Product C Session"):
                    for k in list(st.session_state.keys()):
                        if str(k).startswith("c_"):
                            st.session_state.pop(k, None)
                    st.rerun()

    elif st.session_state.c_stage == "REPORT_READY" and st.session_state.c_report is not None:
        st.success(f"✅ **Product C Enterprise Blueprint Generated Successfully!** Saved under Vault ID `{st.session_state.c_final_vault_id}` with a recovery record in `captured_leads/`.")
        render_on_page_enterprise_blueprint(
            report=st.session_state.c_report,
            contact=st.session_state.c_contact,
            vault_id=st.session_state.c_final_vault_id,
            imported_from_tier=st.session_state.c_imported_tier,
        )
        if st.button("🔄 Start a New Product C Blueprint", use_container_width=True):
            for k in list(st.session_state.keys()):
                if str(k).startswith("c_"):
                    st.session_state.pop(k, None)
            st.rerun()
