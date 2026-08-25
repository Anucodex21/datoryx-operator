"""DATORYX Security Agent - Analyzes security risks, audits code, and recommends protections (LLM-backed)"""
from typing import Dict, Any, List

from ..base import BaseAgent, AgentCapability

SYSTEM_PROMPT = (
    "You are an application security engineer inside DATORYX. You audit real code and system descriptions "
    "for concrete vulnerabilities and risks - grounded in what's actually there, not a generic checklist."
)


class SecurityAgent(BaseAgent):
    """Analyzes security risks, audits code, and recommends protections"""

    def __init__(self, llm: Any = None):
        super().__init__(name="Security", description="Analyzes security risks, audits code, and recommends protections", llm=llm)

    def _register_capabilities(self):
        self.capabilities = [
            AgentCapability("security_audit", "Audit code for vulnerabilities", {"code": "str"}),
            AgentCapability("vulnerability_scan", "Scan dependencies for CVEs", {"dependencies": "list"}),
            AgentCapability("threat_model", "Create threat models", {"system": "str", "assets": "list"}),
            AgentCapability("compliance_check", "Check compliance frameworks", {"framework": "str", "controls": "list"}),
            AgentCapability("incident_response", "Plan incident response", {"incident_type": "str"})
        ]

    def _register_tools(self):
        self.tools = {
            "security_audit": self._security_audit,
            "vulnerability_scan": self._vulnerability_scan,
            "threat_model": self._threat_model,
            "compliance_check": self._compliance_check,
            "incident_response": self._incident_response
        }

    def _security_audit(self, code: str, language: str = "python") -> Dict:
        user_prompt = (
            f"Audit this {language} code for security vulnerabilities:\n\n{code}\n\n"
            'Return JSON: {"vulnerabilities": [{"severity": "critical|high|medium|low", "type": "...", '
            '"detail": "<what and where in the code>"}], "recommendations": ["..."]}'
        )
        result = self._llm_json(SYSTEM_PROMPT, user_prompt)
        vulnerabilities = result.get("vulnerabilities", [])
        score = max(0, 100 - sum({"critical": 30, "high": 20, "medium": 10, "low": 5}.get(v.get("severity"), 10) for v in vulnerabilities))
        grade = "A" if score >= 90 else "B" if score >= 70 else "C" if score >= 50 else "F"
        return {
            "vulnerabilities": vulnerabilities,
            "security_score": score,
            "grade": grade,
            "recommendations": result.get("recommendations", [])
        }

    def _vulnerability_scan(self, dependencies: List[str]) -> Dict:
        user_prompt = (
            f"Given this list of dependency version specifiers, identify which have well-known, publicly "
            f"documented CVEs you are confident about (do not guess or invent CVE numbers): {dependencies}\n\n"
            'Return JSON: {"findings": [{"dependency": "...", "concern": "...", "severity": "...", '
            '"confidence": "high|low"}], "patch_recommendations": ["..."]}'
        )
        result = self._llm_json(SYSTEM_PROMPT, user_prompt)
        findings = result.get("findings", [])
        return {
            "scanned": len(dependencies),
            "vulnerabilities_found": len(findings),
            "findings": findings,
            "patch_recommendations": result.get("patch_recommendations", []),
            "note": "No live CVE database connected - findings reflect general model knowledge only; "
                    "cross-check against a real feed (e.g. OSV, NVD) before acting."
        }

    def _threat_model(self, system_description: str,
                       assets: List[str]) -> Dict:
        user_prompt = (
            f"Build a threat model (e.g. STRIDE-style) for this system: {system_description}\n"
            f"Assets: {assets}\n\n"
            'Return JSON: {"threats": [{"asset": "...", "threat": "...", "likelihood": "...", "impact": "..."}], '
            '"mitigations": ["..."]}'
        )
        result = self._llm_json(SYSTEM_PROMPT, user_prompt)
        return {
            "system": system_description,
            "assets": assets,
            "threats": result.get("threats", []),
            "mitigations": result.get("mitigations", [])
        }

    def _compliance_check(self, framework: str,
                           controls: List[str]) -> Dict:
        user_prompt = (
            f"For the '{framework}' compliance framework, given these implemented controls: {controls}, "
            "identify what's missing.\n\n"
            'Return JSON: {"required_controls": ["..."], "controls_missing": ["..."], '
            '"remediation": ["..."]}'
        )
        result = self._llm_json(SYSTEM_PROMPT, user_prompt)
        required = result.get("required_controls", [])
        missing = result.get("controls_missing", [])
        return {
            "framework": framework,
            "controls_implemented": controls,
            "controls_missing": missing,
            "compliance_rate": (len(controls) / len(required)) if required else 1.0,
            "status": "compliant" if not missing else "non_compliant",
            "remediation": result.get("remediation", missing)
        }

    def _incident_response(self, incident_type: str,
                            severity: str = "medium") -> Dict:
        user_prompt = (
            f"Produce an incident response playbook for a '{incident_type}' incident of '{severity}' severity.\n\n"
            'Return JSON: {"playbook": ["<ordered response steps>"], "communication_plan": "..."}'
        )
        result = self._llm_json(SYSTEM_PROMPT, user_prompt)
        return {
            "incident_type": incident_type,
            "severity": severity,
            "playbook": result.get("playbook", []),
            "escalation": severity in ["critical", "high"],
            "communication_plan": result.get("communication_plan", "")
        }
