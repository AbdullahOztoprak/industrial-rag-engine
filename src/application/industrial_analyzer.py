"""
Industrial Analyzer — Domain-specific analysis for industrial automation queries.

Implements:
- Structured response formatting (problem / root cause / solution / risk / safety)
- Confidence scoring
- Hallucination mitigation
- Domain classification
- Safety warning generation
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from src.domain import (
    ConfidenceLevel,
    IndustrialDomain,
    RiskLevel,
    SafetyWarning,
    SourceAttribution,
)

logger = logging.getLogger(__name__)

# ─── Domain Classification ────────────────────────────────────────────────────

DOMAIN_KEYWORDS: dict[IndustrialDomain, list[str]] = {
    IndustrialDomain.PLC_PROGRAMMING: [
        "plc",
        "ladder logic",
        "structured text",
        "function block",
        "iec 61131",
        "siemens",
        "allen-bradley",
        "codesys",
        "tia portal",
        "scan time",
        "i/o module",
        "digital input",
        "analog output",
    ],
    IndustrialDomain.SCADA_SYSTEMS: [
        "scada",
        "hmi",
        "supervisory",
        "data acquisition",
        "historian",
        "trend",
        "alarm management",
        "opc",
    ],
    IndustrialDomain.BUILDING_AUTOMATION: [
        "building automation",
        "bas",
        "hvac",
        "bacnet",
        "knx",
        "dali",
        "vav",
        "ahu",
        "chiller",
        "boiler",
        "bms",
        "building management",
        "energy efficiency",
    ],
    IndustrialDomain.PREDICTIVE_MAINTENANCE: [
        "predictive maintenance",
        "condition monitoring",
        "vibration",
        "remaining useful life",
        "fault detection",
        "diagnostics",
        "mtbf",
        "mttr",
        "reliability",
        "degradation",
    ],
    IndustrialDomain.INDUSTRIAL_IOT: [
        "industrial iot",
        "iiot",
        "mqtt",
        "opc ua",
        "edge computing",
        "gateway",
        "sensor network",
        "digital twin",
    ],
    IndustrialDomain.MANUFACTURING_EXECUTION: [
        "mes",
        "manufacturing execution",
        "production planning",
        "quality management",
        "sap",
        "erp integration",
    ],
    IndustrialDomain.ALARM_MANAGEMENT: [
        "alarm",
        "alert",
        "notification",
        "isa-18.2",
        "alarm rationalization",
        "alarm flood",
        "shelving",
    ],
    IndustrialDomain.ENERGY_MANAGEMENT: [
        "energy management",
        "power monitoring",
        "load shedding",
        "demand response",
        "iso 50001",
        "energy audit",
    ],
}

# Keywords that trigger safety warnings
SAFETY_KEYWORDS: list[tuple[str, RiskLevel, str]] = [
    (
        "high voltage",
        RiskLevel.CRITICAL,
        "High voltage hazard — qualified personnel only (IEC 60204)",
    ),
    (
        "lockout",
        RiskLevel.CRITICAL,
        "Lockout/Tagout required before maintenance (OSHA 1910.147)",
    ),
    (
        "tagout",
        RiskLevel.CRITICAL,
        "Lockout/Tagout required before maintenance (OSHA 1910.147)",
    ),
    (
        "emergency stop",
        RiskLevel.HIGH,
        "Emergency stop circuits must comply with IEC 60204-1",
    ),
    (
        "safety plc",
        RiskLevel.HIGH,
        "Safety PLCs must meet SIL requirements per IEC 61508",
    ),
    (
        "sil",
        RiskLevel.HIGH,
        "Safety Integrity Level — refer to IEC 61508/61511 standards",
    ),
    (
        "explosive",
        RiskLevel.CRITICAL,
        "Hazardous area classification required (ATEX/IECEx)",
    ),
    (
        "pressure vessel",
        RiskLevel.HIGH,
        "Pressure equipment must comply with local regulations",
    ),
    ("chemical", RiskLevel.HIGH, "Chemical safety assessment required (SDS review)"),
    ("robot", RiskLevel.MEDIUM, "Robotic safety per ISO 10218 / ISO/TS 15066"),
]

# Hallucination indicators
HALLUCINATION_PATTERNS: list[tuple[str, str]] = [
    (r"(?i)i('m| am) (sure|certain|100%)", "Overconfident claim without source"),
    (
        r"(?i)as of (my|the) (last|latest) (update|training)",
        "Potential temporal hallucination",
    ),
    (r"(?i)according to .{0,30}(official|documentation)", "Unverifiable source claim"),
    (r"\d{4}-\d{2}-\d{2}", "Specific date claim — verify accuracy"),
    (r"(?i)always|never|guaranteed", "Absolute statement — may not hold in all cases"),
]


class IndustrialAnalyzer:
    """
    Analyzes industrial queries and enriches LLM responses with
    structured metadata, confidence scoring, and safety warnings.
    """

    def classify_domain(self, query: str) -> IndustrialDomain:
        """
        Classify the industrial domain of a query.

        Uses keyword matching with scoring. Falls back to GENERAL.
        """
        query_lower = query.lower()
        scores: dict[IndustrialDomain, int] = {}

        for domain, keywords in DOMAIN_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in query_lower)
            if score > 0:
                scores[domain] = score

        if not scores:
            return IndustrialDomain.GENERAL

        return max(scores, key=scores.get)  # type: ignore[arg-type]

    def compute_confidence(
        self,
        response_text: str,
        sources: list[SourceAttribution],
        query: str,
    ) -> tuple[ConfidenceLevel, float]:
        """
        Compute a confidence score based on response characteristics.

        Factors:
        - Number and quality of source attributions
        - Response length (too short or too long indicates issues)
        - Presence of hedging language
        - Presence of hallucination indicators
        """
        score = 0.5  # Base score

        # Source quality bonus
        if sources:
            avg_relevance = sum(s.relevance_score for s in sources) / len(sources)
            score += min(avg_relevance * 0.3, 0.3)
        else:
            score -= 0.15

        # Response length analysis
        word_count = len(response_text.split())
        if 50 <= word_count <= 500:
            score += 0.1
        elif word_count < 20:
            score -= 0.2

        # Hedging language penalty
        hedges = ["might", "perhaps", "possibly", "i think", "not sure", "unclear"]
        hedge_count = sum(1 for h in hedges if h in response_text.lower())
        score -= hedge_count * 0.05

        # Hallucination check penalty
        flags = self.detect_hallucination_flags(response_text)
        score -= len(flags) * 0.05

        # Clamp
        score = max(0.0, min(1.0, score))

        # Map to enum
        if score >= 0.8:
            level = ConfidenceLevel.HIGH
        elif score >= 0.5:
            level = ConfidenceLevel.MEDIUM
        elif score >= 0.3:
            level = ConfidenceLevel.LOW
        else:
            level = ConfidenceLevel.UNCERTAIN

        return level, round(score, 3)

    def assess_risk(self, query: str, response_text: str) -> RiskLevel:
        """Assess the risk level based on query and response content."""
        combined = f"{query} {response_text}".lower()

        highest_risk = RiskLevel.LOW
        risk_order = [
            RiskLevel.LOW,
            RiskLevel.MEDIUM,
            RiskLevel.HIGH,
            RiskLevel.CRITICAL,
        ]

        for keyword, risk, _ in SAFETY_KEYWORDS:
            if keyword in combined:
                if risk_order.index(risk) > risk_order.index(highest_risk):
                    highest_risk = risk

        return highest_risk

    def generate_safety_warnings(
        self, query: str, response_text: str
    ) -> list[SafetyWarning]:
        """Generate safety warnings based on content analysis."""
        combined = f"{query} {response_text}".lower()
        warnings: list[SafetyWarning] = []

        for keyword, risk, message in SAFETY_KEYWORDS:
            if keyword in combined:
                warnings.append(
                    SafetyWarning(
                        level=risk,
                        message=message,
                        standard_reference=self._extract_standard(message),
                    )
                )

        return warnings

    def detect_hallucination_flags(self, response_text: str) -> list[str]:
        """Detect potential hallucination indicators in the response."""
        flags: list[str] = []
        for pattern, description in HALLUCINATION_PATTERNS:
            if re.search(pattern, response_text):
                flags.append(description)
        return flags

    def build_system_prompt(self, domain: IndustrialDomain) -> str:
        """
        Build a domain-specific system prompt for the LLM.

        Implements prompt engineering strategy for industrial accuracy.
        """
        base = (
            "You are an expert Industrial AI Knowledge Assistant specializing in "
            "industrial automation, building automation, and manufacturing systems.\n\n"
            "RESPONSE RULES:\n"
            "1. Be precise and technical. Cite specific standards (IEC, ISO, ASHRAE) when relevant.\n"
            "2. If you are unsure, explicitly state your uncertainty level.\n"
            "3. NEVER fabricate specific product models, firmware versions, or configuration values.\n"
            "4. For safety-critical topics, always include relevant safety warnings.\n"
            "5. Structure your response clearly with sections when appropriate.\n"
            "6. When troubleshooting, follow: Problem → Root Cause → Solution → Risk Assessment.\n"
            "7. Reference the provided context documents when available.\n"
            "8. If the question is outside your knowledge, say so rather than guessing.\n\n"
        )

        domain_specifics = {
            IndustrialDomain.PLC_PROGRAMMING: (
                "DOMAIN FOCUS: PLC Programming\n"
                "- Reference IEC 61131-3 programming languages\n"
                "- Consider scan time impacts and real-time constraints\n"
                "- Include safety considerations for machine control\n"
            ),
            IndustrialDomain.SCADA_SYSTEMS: (
                "DOMAIN FOCUS: SCADA Systems\n"
                "- Consider cybersecurity (IEC 62443)\n"
                "- Address alarm management best practices (ISA-18.2)\n"
                "- Include data historian and trending guidance\n"
            ),
            IndustrialDomain.BUILDING_AUTOMATION: (
                "DOMAIN FOCUS: Building Automation\n"
                "- Reference BACnet, KNX, and DALI protocols\n"
                "- Consider energy efficiency (ASHRAE standards)\n"
                "- Address HVAC control strategies\n"
            ),
            IndustrialDomain.PREDICTIVE_MAINTENANCE: (
                "DOMAIN FOCUS: Predictive Maintenance\n"
                "- Include condition monitoring techniques\n"
                "- Reference reliability engineering concepts\n"
                "- Consider sensor selection and data requirements\n"
            ),
        }

        specific = domain_specifics.get(
            domain, "DOMAIN FOCUS: General Industrial Automation\n"
        )
        return base + specific

    @staticmethod
    def _extract_standard(message: str) -> Optional[str]:
        """Extract a standard reference from a safety message."""
        match = re.search(r"(I[ES][CO]\s*[\d\-/]+|OSHA\s*[\d.]+|ATEX|ASHRAE)", message)
        return match.group(0) if match else None
