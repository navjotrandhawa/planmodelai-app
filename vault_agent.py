import os
import re
import json
import hmac
import uuid
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from cryptography.fernet import Fernet, InvalidToken
from qdrant_client import QdrantClient
from qdrant_client.http import models

load_dotenv(override=True)


class PlanModelVaultAgent:
    """
    Zero-Credential Cryptographic Vector Vault & Agentic Analytics Engine.
    - Identity isolation: HMAC-SHA256 blind indexing (zero passwords/emails stored).
    - Payload confidentiality: AES-256 (Fernet) encryption at rest.
    - Semantic intelligence: PII-scrubbed OpenAI embeddings stored in Qdrant Cloud.
    """

    def __init__(self):
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.embed_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        self.chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

        self.qdrant_url = os.getenv("QDRANT_URL")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")
        self.collection_name = os.getenv("QDRANT_COLLECTION", "planmodelai_vault")

        pepper = os.getenv("CRYPTO_PEPPER", "")
        fernet_key = os.getenv("VAULT_FERNET_KEY", "")

        if not pepper or "PASTE-" in pepper:
            raise ValueError("CRYPTO_PEPPER is missing or still has placeholder text in .env")
        if not fernet_key or "PASTE-" in fernet_key:
            raise ValueError("VAULT_FERNET_KEY is missing or still has placeholder text in .env")

        self.pepper_bytes = pepper.encode("utf-8")
        self.cipher = Fernet(fernet_key.encode("utf-8"))

        self.qdrant = QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key,
            check_compatibility=False,
        )

    # ------------------------------------------------------------------
    # 1. CRYPTOGRAPHIC PRIMITIVES (Blind Index + AES-256 + PII Scrubbing)
    # ------------------------------------------------------------------
    @staticmethod
    def generate_client_ref_id() -> str:
        """Generates a human-readable, high-entropy Client Vault Reference ID."""
        token = uuid.uuid4().hex[:8].upper()
        return f"PMAI-{token[:4]}-{token[4:]}"

    def compute_blind_index(self, client_ref_id: str) -> str:
        """Computes a deterministic one-way HMAC-SHA256 blind index for exact lookup."""
        normalized = client_ref_id.strip().upper()
        return hmac.new(
            self.pepper_bytes,
            normalized.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def encrypt_payload(self, report_dict: Dict[str, Any]) -> str:
        """Encrypts full report JSON into AES-256 Fernet ciphertext."""
        raw_bytes = json.dumps(report_dict, ensure_ascii=False).encode("utf-8")
        return self.cipher.encrypt(raw_bytes).decode("utf-8")

    def decrypt_payload(self, ciphertext: str) -> Dict[str, Any]:
        """Decrypts AES-256 Fernet ciphertext back into structured Python dict."""
        try:
            decrypted_bytes = self.cipher.decrypt(ciphertext.encode("utf-8"))
            return json.loads(decrypted_bytes.decode("utf-8"))
        except InvalidToken as exc:
            raise ValueError("Decryption failed: invalid VAULT_FERNET_KEY or corrupted payload.") from exc

    @staticmethod
    def scrub_pii(text: str) -> str:
        """Deterministic regex scrubber to strip emails, phones, and API keys before embedding."""
        text = re.sub(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "[REDACTED_EMAIL]", text)
        text = re.sub(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b", "[REDACTED_PHONE]", text)
        text = re.sub(r"sk-[a-zA-Z0-9_-]{16,}", "[REDACTED_API_KEY]", text)
        return text

    def embed_text(self, text: str) -> List[float]:
        """Generates 1536-dim embedding vector via OpenAI."""
        clean_text = self.scrub_pii(text)
        response = self.openai_client.embeddings.create(
            model=self.embed_model,
            input=clean_text,
        )
        return response.data[0].embedding

    # ------------------------------------------------------------------
    # 2. WRITE PATH: SCRUB, EMBED, ENCRYPT & UPSERT TO QDRANT CLOUD
    # ------------------------------------------------------------------
    def save_blueprint_to_vault(
        self,
        user_inputs: Dict[str, str],
        report_data: Dict[str, Any],
        tier: str = "Tier 1",
        client_ref_id: Optional[str] = None,
    ) -> str:
        """
        Stores an anonymized semantic embedding + indexed analytics metadata +
        AES-256 encrypted full report in Qdrant Cloud.
        Returns the Client Reference ID for zero-password future retrieval.
        """
        ref_id = client_ref_id.strip().upper() if client_ref_id else self.generate_client_ref_id()
        blind_index = self.compute_blind_index(ref_id)

        # Build clean semantic summary (strictly PII-scrubbed)
        semantic_doc = (
            f"Architecture Tier: {tier}. "
            f"Business Goal: {user_inputs.get('goal', '')}. "
            f"Target Users & Workflow: {user_inputs.get('workflow', '')}. "
            f"Data & Compliance Constraints: {user_inputs.get('constraints', '')}. "
            f"Budget & Timeline: {user_inputs.get('budget', '')}. "
            f"Modality: {report_data.get('modality', 'TEXT_REASONING')}. "
            f"Feasibility Grade: {report_data.get('feasibility_grade', 'Grade B')}. "
            f"Executive Summary: {report_data.get('executive_summary', '')}. "
            f"Recommended Cloud Stack: {report_data.get('paid_stack_summary', '')}. "
            f"Recommended Open-Source/VPC Stack: {report_data.get('oss_stack_summary', '')}."
        )
        scrubbed_summary = self.scrub_pii(semantic_doc)
        vector = self.embed_text(scrubbed_summary)

        # Encrypt complete report + original inputs so a DB breach exposes zero secrets
        full_vault_record = {
            "client_ref_id": ref_id,
            "tier": tier,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "user_inputs": user_inputs,
            "report_data": report_data,
        }
        encrypted_blob = self.encrypt_payload(full_vault_record)

        point_id = str(uuid.uuid4())
        payload = {
            # Indexed fields (matching your 5 Qdrant Cloud payload indexes)
            "client_blind_index": blind_index,
            "modality": str(report_data.get("modality", "TEXT_REASONING")),
            "tier": str(tier),
            "feasibility_grade": str(report_data.get("feasibility_grade", "Grade B")),
            "paid_monthly_cost_usd": int(report_data.get("paid_monthly_cost_usd", 0)),
            # Non-indexed fields for analytics & encrypted retrieval
            "oss_monthly_cost_usd": int(report_data.get("oss_monthly_cost_usd", 0)),
            "scrubbed_summary": scrubbed_summary,
            "created_at": full_vault_record["created_at"],
            "encrypted_report_payload": encrypted_blob,
        }

        self.qdrant.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            ],
        )
        return ref_id

    # ------------------------------------------------------------------
    # 3. READ PATH A: ZERO-LOGIN CLIENT VAULT RETRIEVER (5th Module)
    # ------------------------------------------------------------------
    def retrieve_client_reports(self, client_ref_id: str) -> List[Dict[str, Any]]:
        """
        Computes HMAC-SHA256 blind index from user-supplied Client Reference ID,
        filters Qdrant deterministically, and decrypts matching reports in memory.
        """
        blind_index = self.compute_blind_index(client_ref_id)

        records, _ = self.qdrant.scroll(
            collection_name=self.collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="client_blind_index",
                        match=models.MatchValue(value=blind_index),
                    )
                ]
            ),
            limit=20,
            with_payload=True,
            with_vectors=False,
        )

        decrypted_results = []
        for pt in records:
            ciphertext = (pt.payload or {}).get("encrypted_report_payload")
            if ciphertext:
                record = self.decrypt_payload(ciphertext)
                decrypted_results.append(record)

        # Sort newest first
        decrypted_results.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return decrypted_results

    def answer_vault_question(self, client_ref_id: str, question: str) -> Dict[str, Any]:
        """
        Agentic Vault Assistant: Retrieves all decrypted tiers for a Client Reference ID
        and answers architectural or TCO follow-up questions grounded strictly in their vault.
        """
        records = self.retrieve_client_reports(client_ref_id)
        if not records:
            return {
                "found": False,
                "answer": "No architecture blueprints matched that Client Reference ID. Please check the ID format (e.g., PMAI-XXXX-XXXX).",
                "records": [],
            }

        context_blocks = []
        for idx, rec in enumerate(records, 1):
            context_blocks.append(
                f"--- Stored Blueprint #{idx} ({rec.get('tier')} | {rec.get('created_at')}) ---\n"
                f"User Inputs: {json.dumps(rec.get('user_inputs', {}))}\n"
                f"Architecture Report: {json.dumps(rec.get('report_data', {}))}"
            )
        combined_context = "\n\n".join(context_blocks)

        prompt = (
            "You are PlanModelAI's Agentic Vault Architect. A returning client authenticated via their "
            "cryptographic Client Reference ID. Answer their question concisely and accurately using ONLY "
            "their decrypted blueprint records below. Provide concrete cost, AWS sizing, or guardrail details.\n\n"
            f"DECRYPTED CLIENT VAULT RECORDS:\n{combined_context}\n\n"
            f"CLIENT QUESTION: {question}"
        )

        completion = self.openai_client.chat.completions.create(
            model=self.chat_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        return {
            "found": True,
            "answer": completion.choices[0].message.content,
            "records": records,
        }

    # ------------------------------------------------------------------
    # 4. READ PATH B: SEMANTIC DATA ANALYTICS ACROSS ALL SUBMISSIONS
    # ------------------------------------------------------------------
    def run_semantic_analytics(
        self,
        analytics_query: str,
        modality_filter: Optional[str] = None,
        tier_filter: Optional[str] = None,
        max_budget_usd: Optional[int] = None,
        top_k: int = 10,
    ) -> Dict[str, Any]:
        """
        Runs filtered semantic vector search across anonymized blueprints in Qdrant
        and synthesizes executive portfolio & demand insights.
        """
        query_vector = self.embed_text(analytics_query)

        must_conditions = []
        if modality_filter and modality_filter != "ALL":
            must_conditions.append(
                models.FieldCondition(
                    key="modality",
                    match=models.MatchValue(value=modality_filter),
                )
            )
        if tier_filter and tier_filter != "ALL":
            must_conditions.append(
                models.FieldCondition(
                    key="tier",
                    match=models.MatchValue(value=tier_filter),
                )
            )
        if max_budget_usd is not None and max_budget_usd > 0:
            must_conditions.append(
                models.FieldCondition(
                    key="paid_monthly_cost_usd",
                    range=models.Range(lte=max_budget_usd),
                )
            )

        query_filter = models.Filter(must=must_conditions) if must_conditions else None

        search_response = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )

        hits = search_response.points
        if not hits:
            return {
                "match_count": 0,
                "insights": "No stored blueprints matched your semantic query and filter criteria yet.",
                "matches": [],
            }

        matches_summary = []
        for idx, hit in enumerate(hits, 1):
            p = hit.payload or {}
            matches_summary.append({
                "rank": idx,
                "similarity_score": round(hit.score, 3),
                "tier": p.get("tier"),
                "modality": p.get("modality"),
                "feasibility_grade": p.get("feasibility_grade"),
                "paid_monthly_cost_usd": p.get("paid_monthly_cost_usd"),
                "oss_monthly_cost_usd": p.get("oss_monthly_cost_usd"),
                "scrubbed_summary": p.get("scrubbed_summary"),
            })

        analytics_prompt = (
            "You are a Principal AI Solutions Architect analyzing anonymized enterprise demand data from "
            "PlanModelAI's Qdrant vector database. Synthesize key trends, average cost patterns, common "
            "architectural bottlenecks, and recommendations based on the retrieved matches below.\n\n"
            f"ANALYST QUERY: {analytics_query}\n\n"
            f"RETRIEVED ANONYMIZED BLUEPRINTS:\n{json.dumps(matches_summary, indent=2)}"
        )

        completion = self.openai_client.chat.completions.create(
            model=self.chat_model,
            messages=[{"role": "user", "content": analytics_prompt}],
            temperature=0.2,
        )

        return {
            "match_count": len(matches_summary),
            "insights": completion.choices[0].message.content,
            "matches": matches_summary,
        }