"""Model instructions for typed read and bounded-answer owners."""

from __future__ import annotations

from agnostic_market.agents._shared_prompt import compose_shared_context
from agnostic_market.commerce.catalog import CatalogProductSet
from agnostic_market.dtos.orchestration import AnswerQuestion
from agnostic_market.dtos.state import PolicyContext

ORDER_TARGET_PROPOSAL_PROMPT = """Classify only the order references stated in the caller's message.
Return single for one intended order, plural for multiple intended orders, alternative when the
caller presents choices, and ambiguous when a reference is merely quoted or disclaimed, or when
fragmentary, conflicting, or unresolved wording does not establish an executable target. A clear
correction keeps only the corrected-to reference. Preserve every observed order reference on
non-single arms; do not select one alternative or infer an order from conversational context.
Normalize a clear labelled numeric reference to ORD-<digits>."""


def compose_catalog_response_prompt(
    display_name: str,
    policy: PolicyContext,
    result: CatalogProductSet,
) -> str:
    """Bound a product answer to one current catalog lookup."""

    if result.products:
        catalog_facts = "\n".join(
            f"- {product.name}; SKU {product.sku}; price ${product.price_usd:.2f}"
            for product in result.products
        )
        lookup_instruction = "Answer using only the matching catalog facts below."
    else:
        catalog_facts = "- No matching catalog products."
        lookup_instruction = (
            "Say that no catalog product matched the request. Do not claim that other products "
            "match or list products absent from the result."
        )
    return "\n".join(
        (
            compose_shared_context(display_name, policy),
            "",
            "You are the product-catalog response owner. This is a read-only answer.",
            lookup_instruction,
            "Do not invent products, prices, SKUs, stock, shipping, or availability.",
            "Keep the spoken answer to one or two short sentences.",
            "",
            "Live catalog result:",
            catalog_facts,
        )
    )


def compose_answer_response_prompt(
    display_name: str,
    policy: PolicyContext,
    request: AnswerQuestion,
) -> str:
    """Compose one closed instruction branch for the bounded question owner."""

    if request.topic == "policy":
        topic_instruction = (
            "For a self-contained policy question, answer only from the approved merchant policy "
            "facts above. If those facts do not cover the detail, say that the detail is not "
            "available. Do not use general model knowledge to fill a merchant-policy gap."
        )
    else:
        topic_instruction = (
            "For a self-contained general question, you may give a low-risk explanation from "
            "general knowledge, but do not make claims about this merchant, an account, an order, "
            "inventory, a transfer, or whether any commerce effect happened."
        )
    return "\n".join(
        (
            compose_shared_context(display_name, policy),
            "",
            "You are the terminal, tool-incapable bounded-question response owner.",
            topic_instruction,
            "If a question is context-dependent and requires a live merchant, account, order, "
            "inventory, transfer, or commerce-effect owner, unsupported takes precedence.",
            "Return decision=clarify for an incomplete or context-dependent question.",
            "Return decision=unsupported for merchant-specific, account, order, inventory, "
            "transfer, or commerce-effect questions that need a live owner.",
            "The shared guidance about work being handled elsewhere does not apply to this "
            "terminal owner.",
            "Never promise to check, handle, transfer, or follow up.",
            "Return decision=answer only when the answer is within the selected topic boundary.",
            "Keep an answer to one or two short sentences.",
        )
    )
