"""Verify configuration is loaded correctly."""
import sys
sys.path.insert(0, '..')

from research_system.config import (
    MAX_EVIDENCE_STEPS,
    MAX_AGENT_STEPS,
    MIN_SECTION_LENGTH_WORDS,
    MIN_REPORT_SECTIONS,
    MAX_OBSERVATION_CHARS,
)

print("=" * 60)
print("CONFIGURATION VERIFICATION")
print("=" * 60)
print(f"[OK] MAX_EVIDENCE_STEPS = {MAX_EVIDENCE_STEPS}")
print(f"[OK] MAX_AGENT_STEPS = {MAX_AGENT_STEPS}")
print(f"[OK] MAX_OBSERVATION_CHARS = {MAX_OBSERVATION_CHARS}")
print(f"[OK] MIN_SECTION_LENGTH_WORDS = {MIN_SECTION_LENGTH_WORDS}")
print(f"[OK] MIN_REPORT_SECTIONS = {MIN_REPORT_SECTIONS}")
print("=" * 60)

# Verify expected values
assert MAX_EVIDENCE_STEPS == 150, f"Expected MAX_EVIDENCE_STEPS=150, got {MAX_EVIDENCE_STEPS}"
assert MAX_AGENT_STEPS == 100, f"Expected MAX_AGENT_STEPS=100, got {MAX_AGENT_STEPS}"
assert MIN_SECTION_LENGTH_WORDS == 400, f"Expected MIN_SECTION_LENGTH_WORDS=400, got {MIN_SECTION_LENGTH_WORDS}"
assert MIN_REPORT_SECTIONS == 6, f"Expected MIN_REPORT_SECTIONS=6, got {MIN_REPORT_SECTIONS}"

print("[SUCCESS] All configuration values verified successfully!")
