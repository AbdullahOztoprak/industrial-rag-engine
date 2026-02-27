"""
Unit tests for IndustrialAnalyzer.

Tests domain classification, confidence scoring, risk assessment,
safety warnings, and hallucination detection.
"""

import pytest

from src.application.industrial_analyzer import IndustrialAnalyzer
from src.domain import (
    ConfidenceLevel,
    IndustrialDomain,
    RiskLevel,
    SourceAttribution,
)


@pytest.fixture
def analyzer():
    """Create an IndustrialAnalyzer instance."""
    return IndustrialAnalyzer()


class TestDomainClassification:
    """Tests for query domain classification."""

    def test_plc_classification(self, analyzer):
        query = "How do I program ladder logic in a PLC?"
        assert analyzer.classify_domain(query) == IndustrialDomain.PLC_PROGRAMMING

    def test_scada_classification(self, analyzer):
        query = "What are best practices for SCADA alarm management?"
        assert analyzer.classify_domain(query) == IndustrialDomain.SCADA_SYSTEMS

    def test_building_automation_classification(self, analyzer):
        query = "How do I configure BACnet communication for the BMS?"
        assert analyzer.classify_domain(query) == IndustrialDomain.BUILDING_AUTOMATION

    def test_predictive_maintenance_classification(self, analyzer):
        query = "How to set up vibration monitoring for predictive maintenance?"
        assert (
            analyzer.classify_domain(query) == IndustrialDomain.PREDICTIVE_MAINTENANCE
        )

    def test_iot_classification(self, analyzer):
        query = "How to integrate OPC UA with an industrial IoT gateway?"
        assert analyzer.classify_domain(query) == IndustrialDomain.INDUSTRIAL_IOT

    def test_mes_classification(self, analyzer):
        query = "How to design a manufacturing execution system with SAP integration?"
        assert (
            analyzer.classify_domain(query) == IndustrialDomain.MANUFACTURING_EXECUTION
        )

    def test_alarm_classification(self, analyzer):
        query = "How to handle alarm flood situations per ISA-18.2?"
        assert analyzer.classify_domain(query) == IndustrialDomain.ALARM_MANAGEMENT

    def test_energy_classification(self, analyzer):
        query = "Implement ISO 50001 energy management in a factory"
        assert analyzer.classify_domain(query) == IndustrialDomain.ENERGY_MANAGEMENT

    def test_general_fallback(self, analyzer):
        query = "What is the meaning of life?"
        assert analyzer.classify_domain(query) == IndustrialDomain.GENERAL

    def test_case_insensitive(self, analyzer):
        query = "SCADA SYSTEM ALARM MANAGEMENT"
        assert analyzer.classify_domain(query) == IndustrialDomain.SCADA_SYSTEMS


class TestConfidenceScoring:
    """Tests for confidence score computation."""

    def test_high_confidence_with_sources(self, analyzer):
        sources = [
            SourceAttribution(document="doc1", relevance_score=0.9, excerpt="..."),
            SourceAttribution(document="doc2", relevance_score=0.85, excerpt="..."),
        ]
        response = (
            "This is a detailed technical explanation about PLC control loops " * 5
        )
        level, score = analyzer.compute_confidence(response, sources, "PLC question")
        assert score > 0.5
        assert level in [ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM]

    def test_low_confidence_without_sources(self, analyzer):
        level, score = analyzer.compute_confidence(
            "Not sure about this.", [], "complex question"
        )
        assert score < 0.6

    def test_very_short_response_penalized(self, analyzer):
        level, score = analyzer.compute_confidence("Yes.", [], "question")
        assert score < 0.5

    def test_hedging_language_penalized(self, analyzer):
        _, score_hedge = analyzer.compute_confidence(
            "I think maybe perhaps this might possibly work.", [], "q"
        )
        _, score_confident = analyzer.compute_confidence(
            "The PID controller should be configured with Kp=2.0, Ti=5s.", [], "q"
        )
        assert score_confident > score_hedge

    def test_confidence_clamped_to_range(self, analyzer):
        # Test with extreme inputs
        _, score = analyzer.compute_confidence("x", [], "q")
        assert 0.0 <= score <= 1.0


class TestRiskAssessment:
    """Tests for risk level assessment."""

    def test_low_risk(self, analyzer):
        risk = analyzer.assess_risk("What is BACnet?", "BACnet is a protocol...")
        assert risk == RiskLevel.LOW

    def test_high_risk_safety_plc(self, analyzer):
        risk = analyzer.assess_risk(
            "Configure safety PLC", "Safety PLC configuration..."
        )
        assert risk in [RiskLevel.HIGH, RiskLevel.CRITICAL]

    def test_critical_risk_emergency_stop(self, analyzer):
        risk = analyzer.assess_risk(
            "emergency stop circuit", "The emergency stop must..."
        )
        assert risk in [RiskLevel.HIGH, RiskLevel.CRITICAL]

    def test_critical_risk_high_voltage(self, analyzer):
        risk = analyzer.assess_risk(
            "working with high voltage panels", "High voltage equipment requires..."
        )
        assert risk == RiskLevel.CRITICAL


class TestSafetyWarnings:
    """Tests for safety warning generation."""

    def test_no_warnings_for_safe_query(self, analyzer):
        warnings = analyzer.generate_safety_warnings(
            "What is BACnet?", "BACnet is a protocol..."
        )
        assert len(warnings) == 0

    def test_warning_for_emergency_stop(self, analyzer):
        warnings = analyzer.generate_safety_warnings(
            "emergency stop design", "The emergency stop circuit should..."
        )
        assert len(warnings) > 0
        assert any(w.level in [RiskLevel.HIGH, RiskLevel.CRITICAL] for w in warnings)

    def test_warning_for_lockout_tagout(self, analyzer):
        warnings = analyzer.generate_safety_warnings(
            "lockout tagout procedures",
            "Before maintenance, lockout/tagout must be performed...",
        )
        assert len(warnings) > 0

    def test_warning_has_standard_reference(self, analyzer):
        warnings = analyzer.generate_safety_warnings(
            "safety plc configuration",
            "Safety PLCs require SIL assessment per IEC 61508",
        )
        # At least one warning should have a standard reference
        has_reference = any(w.standard_reference is not None for w in warnings)
        assert has_reference


class TestHallucinationDetection:
    """Tests for hallucination flag detection."""

    def test_no_flags_for_clean_text(self, analyzer):
        flags = analyzer.detect_hallucination_flags(
            "PID control uses proportional, integral, and derivative terms."
        )
        assert len(flags) == 0

    def test_flag_overconfident_claim(self, analyzer):
        flags = analyzer.detect_hallucination_flags(
            "I'm 100% certain this is the correct configuration."
        )
        assert len(flags) > 0

    def test_flag_absolute_statement(self, analyzer):
        flags = analyzer.detect_hallucination_flags(
            "This method always works and is guaranteed to succeed."
        )
        assert len(flags) > 0

    def test_flag_temporal_hallucination(self, analyzer):
        flags = analyzer.detect_hallucination_flags(
            "As of my last training update, the standard requires..."
        )
        assert len(flags) > 0


class TestSystemPromptGeneration:
    """Tests for domain-specific system prompt generation."""

    def test_plc_prompt_contains_iec(self, analyzer):
        prompt = analyzer.build_system_prompt(IndustrialDomain.PLC_PROGRAMMING)
        assert "IEC 61131" in prompt

    def test_scada_prompt_contains_cybersecurity(self, analyzer):
        prompt = analyzer.build_system_prompt(IndustrialDomain.SCADA_SYSTEMS)
        assert "cybersecurity" in prompt.lower() or "62443" in prompt

    def test_bas_prompt_contains_bacnet(self, analyzer):
        prompt = analyzer.build_system_prompt(IndustrialDomain.BUILDING_AUTOMATION)
        assert "BACnet" in prompt

    def test_general_prompt_has_base_rules(self, analyzer):
        prompt = analyzer.build_system_prompt(IndustrialDomain.GENERAL)
        assert "NEVER fabricate" in prompt
        assert "safety" in prompt.lower()
