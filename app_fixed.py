#!/usr/bin/env python3
"""
SecureTest AI v2 — Clean Architecture
=====================================
FEATURE → OWASP VULNERABILITIES → CVE/CVSS METRICS → PRIORITIZED TEST CASES

Pipeline:
  1. User inputs feature description
  2. LLM identifies applicable OWASP Top 10 vulnerabilities (A01-A10:2021)
  3. For each vulnerability: extract relevant CVEs, CVSS scores, CWE
  4. Prioritize test cases by CVSS, then generate with LLM
  5. Output: prioritized security test cases with metrics

Collections:
  - securecode: coding patterns + OWASP mappings
  - cve: CVE database for vulnerability research
  - cvss_dataset: CVSS scores for priority calculation
  - cisa_csaf: KEV (Known Exploited Vulnerabilities)
"""

import streamlit as st
import requests
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import re
import os
from typing import Optional, List, Dict, Tuple
import warnings
import time
import traceback
from datasets import load_dataset
import uuid

warnings.filterwarnings("ignore")

# =============================================================================
# PAGE CONFIG & STYLING
# =============================================================================
st.set_page_config(
    page_title="SecureTest AI v2 | OWASP-Based Test Generation",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
:root {
    --critical: #dc3545;
    --high: #fd7e14;
    --medium: #ffc107;
    --low: #28a745;
    --rag: #6610f2;
}
#MainMenu{visibility:hidden;} footer{visibility:hidden;}
.metric-badge {
    padding: 4px 10px; border-radius: 6px; font-weight: 600; 
    display: inline-block; margin: 2px; font-size: 12px;
}
.badge-p1 { background: var(--critical); color: white; }
.badge-p2 { background: var(--high); color: white; }
.badge-p3 { background: var(--medium); color: black; }
.badge-p4 { background: var(--low); color: white; }
.metric-card {
    background: #f8f9fa; padding: 12px; border-radius: 8px;
    border-left: 4px solid var(--rag); margin: 6px 0;
}
.threat-section { border: 1px solid #e0e0e0; padding: 12px; border-radius: 8px; margin: 8px 0; }
.step-item { background: #f5f5f5; padding: 8px; margin: 4px 0; border-radius: 4px; }
.priority-banner {
    padding: 10px 14px; border-radius: 6px; margin: 4px 0; font-weight: 600;
}
.p1-banner { background: #dc354530; color: #721c24; border-left: 4px solid var(--critical); }
.p2-banner { background: #fd7e1430; color: #856404; border-left: 4px solid var(--high); }
.p3-banner { background: #ffc10730; color: #856404; border-left: 4px solid var(--medium); }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CONFIGURATION
# =============================================================================
OLLAMA_BASE_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "ministral-3:3b")

# =============================================================================
# OLLAMA INTEGRATION
# =============================================================================
def check_ollama_ready() -> Tuple[bool, str]:
    """Check if Ollama is running and model is available."""
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if r.status_code != 200:
            return False, "Ollama not responding (HTTP error)"
        
        models = [m.get("name", "") for m in r.json().get("models", [])]
        if not models:
            return False, "No models loaded in Ollama"
        
        if OLLAMA_MODEL in models:
            return True, f"✅ {OLLAMA_MODEL}"
        
        return True, f"⚠️  Using {models[0]} (requested: {OLLAMA_MODEL})"
    except requests.exceptions.ConnectionError:
        return False, "Ollama not running — execute: ollama serve"
    except Exception as e:
        return False, str(e)


def call_llm(prompt: str, temperature: float = 0.2, max_tokens: int = 3000) -> str:
    """Call LLM with retry logic, debug output, and response validation."""
    for attempt in range(3):
        try:
            r = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens,
                        "num_gpu": 99,
                        "num_thread": 8,
                        "top_p": 0.9,
                    },
                },
                timeout=180,
            )
            if r.status_code == 200:
                response_text = r.json().get("response", "").strip()
                if response_text:
                    # Log first 200 chars for debugging
                    import sys
                    print(f"[LLM DEBUG] Response received ({len(response_text)} chars): {response_text[:200]}...", file=sys.stderr)
                    return response_text
                else:
                    print(f"[LLM DEBUG] Empty response on attempt {attempt + 1}", file=sys.stderr)
                    if attempt == 2:
                        st.warning(f"⚠️  LLM returned empty response after {attempt + 1} attempts")
                    else:
                        time.sleep(2 ** attempt)
                    continue
        except requests.exceptions.Timeout:
            print(f"[LLM DEBUG] Timeout on attempt {attempt + 1}", file=sys.stderr)
            if attempt == 2:
                st.error(f"❌ LLM call timed out after {attempt + 1} attempts (180s timeout)")
            else:
                time.sleep(2 ** attempt)
        except Exception as e:
            print(f"[LLM DEBUG] Exception on attempt {attempt + 1}: {str(e)}", file=sys.stderr)
            if attempt == 2:
                st.error(f"❌ LLM call failed: {str(e)}")
            else:
                time.sleep(2 ** attempt)
    
    print("[LLM DEBUG] All LLM attempts exhausted", file=sys.stderr)
    return ""


# =============================================================================
# JSON PARSING
# =============================================================================
def parse_json_robust(text: str) -> Optional[Dict]:
    """Extract and parse JSON from text."""
    if not text:
        return None
    
    # Try direct parse
    try:
        return json.loads(text)
    except:
        pass
    
    # Find JSON block
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return None
    
    try:
        return json.loads(text[start:end])
    except:
        pass
    
    # Try json_repair if available
    try:
        from json_repair import repair_json
        return repair_json(text[start:end], return_objects=True)
    except:
        return None


def parse_json_array(text: str) -> Optional[List[Dict]]:
    """Extract and parse JSON array from text, handling markdown formatting."""
    if not text:
        return None
    
    # Strip markdown code blocks (```json ... ```)
    text = re.sub(r'^```json\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s*```$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^```\s*', '', text, flags=re.MULTILINE)
    
    try:
        return json.loads(text)
    except:
        pass
    
    start = text.find("[")
    end = text.rfind("]") + 1
    if start == -1 or end == 0:
        return None
    
    try:
        return json.loads(text[start:end])
    except:
        pass
    
    # Try object with array key
    obj = parse_json_robust(text)
    if obj and isinstance(obj, dict):
        for key in ("test_cases", "tests", "results", "items"):
            if key in obj and isinstance(obj[key], list):
                return obj[key]
    
    return None


# =============================================================================
# PHASE 1: OWASP VULNERABILITY IDENTIFICATION
# =============================================================================
def identify_owasp_vulnerabilities(feature_desc: str) -> List[Dict]:
    """
    LLM identifies OWASP Top 10:2021 vulnerabilities applicable to the feature.
    Returns list of vulnerabilities with OWASP code, description, and severity.
    """
    # Try multiple prompts with different formats to maximize parsing success
    prompts = [
        # Format 1: Strict JSON array
        (
            "You are a security architect. Analyze the feature and identify OWASP Top 10:2021 vulnerabilities.\n\n"
            "Output ONLY valid JSON array with this exact structure:\n"
            '[{"owasp": "A01:2021", "name": "Broken Access Control", '
            '"description": "...", "severity": "High", "affected_component": "..."}]\n\n'
            f"Feature: {feature_desc}\n\n"
            "For each relevant OWASP code (A01-A10:2021), provide name, brief description, severity (Low/Medium/High/Critical), "
            "and affected component. Output ONLY the JSON array, nothing else.\n\nJSON array:"
        ),
        # Format 2: Alternative wording with numbered list output
        (
            "You are a security expert. List 3-5 OWASP Top 10:2021 vulnerabilities relevant to this feature.\n\n"
            f"Feature: {feature_desc}\n\n"
            "For each vulnerability, output a JSON object on a new line:\n"
            '{"owasp": "A01:2021", "name": "Broken Access Control", "description": "brief description", "severity": "High", "affected_component": "component name"}\n\n'
            "Output only JSON objects, one per line:"
        ),
    ]
    
    vulnerabilities = None
    for attempt_idx, prompt in enumerate(prompts):
        # Increase temperature with each attempt for more creative parsing
        temp = 0.6 + (attempt_idx * 0.15)  # 0.6, 0.75, etc.
        response = call_llm(prompt, temperature=min(temp, 0.85), max_tokens=2000)
        
        if response:
            # Try parsing as array first
            vulnerabilities = parse_json_array(response)
            if vulnerabilities and len(vulnerabilities) > 0:
                break
            
            # Try parsing line-by-line for Format 2
            if not vulnerabilities:
                line_vulns = []
                for line in response.split('\n'):
                    if line.strip().startswith('{'):
                        try:
                            obj = json.loads(line.strip())
                            if 'owasp' in obj:
                                line_vulns.append(obj)
                        except:
                            pass
                if line_vulns:
                    vulnerabilities = line_vulns
                    break
    
    if vulnerabilities and len(vulnerabilities) > 0:
        st.info(f"✅ LLM identified vulnerabilities: {', '.join([v.get('owasp', 'Unknown') for v in vulnerabilities])}")
        return vulnerabilities
    
    # Only use hardcoded fallback if LLM completely fails all parsing attempts
    st.warning("⚠️  LLM failed to identify vulnerabilities. Using fallback OWASP list.")
    st.info("💡 **DEBUG**: Try these troubleshooting steps:\n"
            "1. Check that Ollama is responding: `curl http://localhost:11434/api/tags`\n"
            "2. Test the model directly: `ollama run mistral \"Hello\"`\n"
            "3. Check for JSON output: `ollama run mistral \"output only valid JSON: {\\\"test\\\": \\\"value\\\"}\"`\n"
            "4. Try a different model: `export OLLAMA_MODEL=llama2` then restart")
    return [
        {
            "owasp": "A01:2021",
            "name": "Broken Access Control",
            "description": "Unauthorized access to resources",
            "severity": "High",
            "affected_component": "Authorization layer",
        },
        {
            "owasp": "A03:2021",
            "name": "Injection",
            "description": "Injection of malicious code/data",
            "severity": "Critical",
            "affected_component": "Input handlers",
        },
        {
            "owasp": "A07:2021",
            "name": "Identification and Authentication Failures",
            "description": "Broken authentication mechanisms",
            "severity": "High",
            "affected_component": "Authentication",
        },
    ]


# =============================================================================
# PHASE 2: CVE & CVSS RETRIEVAL (RAG)
# =============================================================================
def load_rag_collections():
    """Load RAG collections: ChromaDB for patterns + Hugging Face for CVEs."""
    try:
        from chromadb import PersistentClient
        
        client = PersistentClient(path="./chroma_db")
        collections = {
            "securecode": client.get_or_create_collection("securecode"),
            "cve": None,
            "cvss": None,
            "cisa": None,
        }
    except Exception as e:
        st.warning(f"ChromaDB not available: {e}")
        collections = {
            "securecode": None,
            "cve": None,
            "cvss": None,
            "cisa": None,
        }
    
    # Load Hugging Face CVE dataset
    try:
        @st.cache_resource
        def load_hf_cve_dataset():
            return load_dataset("Nitish-Garikoti/All-CVE-Chat-MultiTurn-1999-2025-Dataset")
        
        collections["cve_hf"] = load_hf_cve_dataset()
    except Exception as e:
        st.warning(f"Hugging Face CVE dataset not available: {e}")
        collections["cve_hf"] = None
    
    return collections


@st.cache_data(ttl=3600)
def _search_nvd_api_cached(keyword: str, api_key: str = "") -> List[Dict]:
    """
    Cached NVD API search to avoid redundant queries within 1 hour.
    Reduces API rate limit impact for multi-vulnerability analysis.
    """
    results = []
    
    try:
        for attempt in range(3):
            try:
                url = f"https://services.nvd.nist.gov/rest/json/cves/2.0?keywordSearch={keyword}&resultsPerPage=10"
                if api_key:
                    url += f"&apiKey={api_key}"
                
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    for vuln in data.get("vulnerabilities", [])[:10]:
                        cve = vuln.get("cve", {})
                        cve_id = cve.get("id", "")
                        cvss_score = 0
                        
                        # Extract CVSS score from metrics
                        metrics = cve.get("metrics", {})
                        for version in ["cvssMetricV31", "cvssMetricV3", "cvssMetricV2"]:
                            if version in metrics and metrics[version]:
                                cvss_score = metrics[version][0].get("cvssData", {}).get("baseScore", 0)
                                if cvss_score:
                                    break
                        
                        # Extract CWE from weaknesses field
                        cwe_list = []
                        weaknesses = cve.get("weaknesses", [])
                        for weakness in weaknesses:
                            for desc in weakness.get("description", []):
                                cwe_val = desc.get("value", "")
                                if cwe_val and cwe_val.startswith("CWE-"):
                                    cwe_list.append(cwe_val)
                        cwe_str = ", ".join(cwe_list[:3]) if cwe_list else ""  # Top 3 CWEs
                        
                        if cve_id and cvss_score and cvss_score > 0:
                            severity = "Critical" if cvss_score >= 9 else "High" if cvss_score >= 7 else "Medium" if cvss_score >= 4 else "Low"
                            results.append({
                                "cve_id": cve_id,
                                "cvss": float(cvss_score),
                                "severity": severity,
                                "cwe": cwe_str,  # Now properly populated
                                "year": cve.get("published", "")[:4],
                                "description": cve.get("descriptions", [{}])[0].get("value", "")[:100],
                            })
                    return results[:10]
                    
                elif response.status_code == 429:
                    # Rate limited - exponential backoff
                    wait_time = 2 ** attempt + 1
                    time.sleep(wait_time)
                    continue
                else:
                    break
            except Exception as e:
                if attempt == 2:
                    break
                time.sleep(2 ** attempt)
    except:
        pass
    
    return results


def search_cves_for_vulnerability(owasp_vuln: Dict, collections: Dict) -> List[Dict]:
    """
    Search NVD (National Vulnerability Database) for CVEs with real CVSS scores.
    Uses session-level caching to reduce API rate limiting impact.
    Falls back to HF dataset if NVD is unavailable.
    Note: Get free NVD API key from https://nvd.nist.gov/developers/request-an-api-key
    Set as NIST_API_KEY environment variable to avoid rate limiting.
    
    Returns:
        List of CVEs with: cve_id, cvss, severity, cwe, year, description
    """
    results = []
    
    # Try NVD API first (real CVSS scores) - now cached
    try:
        keywords = [
            owasp_vuln["name"].replace(" ", "%20"),
        ]
        
        api_key = os.getenv("NIST_API_KEY", "")
        
        for keyword in keywords[:2]:
            if len(results) >= 5:
                break
            
            # Use cached NVD API search to reduce rate limiting impact
            nvd_results = _search_nvd_api_cached(keyword, api_key)
            results.extend(nvd_results)
        
        if len(results) > 0:
            return results[:10]
    except:
        pass
    
    # Fallback: use HF dataset
    if not collections.get("cve_hf"):
        return results
    
    try:
        dataset = collections["cve_hf"]
        if isinstance(dataset, dict):
            split_name = list(dataset.keys())[0]
            data = dataset[split_name]
        else:
            data = dataset
        
        for i, record in enumerate(data):
            if i > 200 or len(results) >= 5:
                break
            
            cve_id = record.get("cve_id") or record.get("CVE") or ""
            cvss_score = _extract_cvss_from_record(record)
            cwe = _extract_cwe_from_record(record)
            
            if cve_id and cvss_score and cvss_score > 0:
                results.append({
                    "cve_id": cve_id,
                    "cvss": cvss_score,
                    "severity": record.get("severity", "Unknown"),
                    "cwe": cwe if cwe else "Unknown",
                    "year": record.get("published_date", ""),
                    "description": (record.get("description") or record.get("summary") or "")[:100],
                })
    except:
        pass
    
    return results[:10]


def _extract_cvss_from_record(record: Dict) -> float:
    """Extract CVSS score from a CVE record, trying multiple field names."""
    # Try common CVSS field names in order of preference
    cvss_fields = [
        "cvss_score", "cvss", "cvss_v3_score", "cvss_v2_score",
        "cvss_v3.1_score", "score", "base_score", "rating"
    ]
    
    for field in cvss_fields:
        if field in record and record[field]:
            try:
                val = float(record[field])
                if val > 0:  # Return first positive CVSS found
                    return val
            except (ValueError, TypeError):
                continue
    
    return 0.0


def _extract_cwe_from_record(record: Dict) -> str:
    """Extract CWE from a CVE record, trying multiple field names."""
    # Try common CWE field names
    cwe_fields = ["cwe", "cwe_id", "weaknesses", "weakness"]
    
    for field in cwe_fields:
        if field in record and record[field]:
            cwe_val = record[field]
            # Handle list format (e.g., ["CWE-79", "CWE-89"])
            if isinstance(cwe_val, list):
                # Filter for CWE entries only
                cwe_list = [str(item) for item in cwe_val if "CWE" in str(item)]
                if cwe_list:
                    return ", ".join(cwe_list[:3])  # Return top 3
            else:
                # Handle string format
                cwe_str = str(cwe_val).strip()
                if cwe_str and "CWE" in cwe_str:
                    return cwe_str[:100]  # Limit length
    
    return ""


def search_patterns_for_vulnerability(owasp_vuln: Dict, collections: Dict) -> List[Dict]:
    """Search for secure coding patterns related to OWASP vulnerability."""
    patterns = []
    
    if not collections.get("securecode"):
        return patterns
    
    try:
        query = f"{owasp_vuln['name']} mitigation pattern"
        hits = collections["securecode"].query(
            query_texts=[query],
            n_results=3,
            include=["metadatas", "documents"],
        )
        
        for content, metadata in zip(
            hits.get("documents", [[]]), 
            hits.get("metadatas", [[]])
        ):
            for doc, meta in zip(content, metadata):
                patterns.append({
                    "pattern": doc[:200],
                    "owasp": meta.get("owasp_code", ""),
                    "language": meta.get("language", ""),
                })
    except:
        pass
    
    return patterns


def calculate_priority(cvss: float) -> str:
    """Convert CVSS score to priority level."""
    if cvss >= 9.0:
        return "P1"
    elif cvss >= 7.0:
        return "P2"
    elif cvss >= 4.0:
        return "P3"
    return "P4"


# =============================================================================
# PHASE 3: TEST CASE GENERATION
# =============================================================================
def generate_test_cases_for_vulnerability(
    feature_desc: str,
    owasp_vuln: Dict,
    cves: List[Dict],
    patterns: List[Dict],
    test_case_counter: int = 1,
) -> Tuple[List[Dict], int]:
    """
    Generate test cases for an OWASP vulnerability using LLM.
    Prioritizes by CVSS score from CVE data.
    Returns test cases and updated counter for unique IDs.
    Uses multiple prompt formats with increasing temperatures for better JSON parsing.
    """
    # Sort CVEs by CVSS
    cves_sorted = sorted(cves, key=lambda x: x.get("cvss", 0), reverse=True)
    
    # Build context for LLM
    cve_context = "\n".join([
        f"- {cve['cve_id']}: CVSS {cve['cvss']}, {cve['description']}"
        for cve in cves_sorted[:5]
    ]) or "No relevant CVEs found"
    
    pattern_context = "\n".join([
        f"- {p['pattern']}"
        for p in patterns[:3]
    ]) or "No patterns available"
    
    # Try multiple prompt formats with increasing temperatures
    prompts = [
        # Format 1: Standard strict JSON
        (
            "You are a security test engineer. Generate detailed test cases for the vulnerability.\n\n"
            f"OWASP: {owasp_vuln['owasp']} — {owasp_vuln['name']}\n"
            f"Feature: {feature_desc}\n"
            f"Severity: {owasp_vuln['severity']}\n\n"
            f"Related CVEs:\n{cve_context}\n\n"
            f"Mitigation patterns:\n{pattern_context}\n\n"
            "Output ONLY a valid JSON array. No markdown, no explanatory text. Start with [ and end with ].\n"
            'Structure: [{"id": 1, "title": "...", "description": "...", '
            '"steps": ["step 1", "step 2", "step 3"], '
            '"expected_result": "...", "metric": {"cvss": 7.5, "cwe": "CWE-123"}}]\n\n'
            "Generate 5-10 comprehensive test cases.\n\nJSON array:"
        ),
        # Format 2: Simpler structure with explicit line breaks
        (
            "You are a security test engineer. Generate test cases for this vulnerability:\n\n"
            f"{owasp_vuln['owasp']}: {owasp_vuln['name']}\n"
            f"Feature: {feature_desc}\n\n"
            "Output ONLY JSON objects, one per line. Each line must be a complete JSON object:\n"
            '{"id": 1, "title": "Test case title", "description": "...", '
            '"steps": ["step 1", "step 2", "step 3"], "expected_result": "...", "metric": {"cvss": 7.5}}\n\n'
            "Generate 5-10 test cases, outputting one JSON object per line:"
        ),
    ]
    
    test_cases = []
    for prompt_idx, prompt_text in enumerate(prompts):
        if test_cases and len(test_cases) > 0:
            break  # Stop if we got results
        
        # Increase temperature with each attempt
        temp = 0.7 + (prompt_idx * 0.1)  # 0.7, 0.8, etc.
        temp = min(temp, 0.9)
        
        response = call_llm(prompt_text, temperature=temp, max_tokens=3000)
        
        if response:
            # Try parsing as array first
            test_cases = parse_json_array(response) or []
            if test_cases and len(test_cases) > 0:
                break
            
            # Try parsing line-by-line for Format 2
            if not test_cases:
                line_cases = []
                for line in response.split('\n'):
                    if line.strip().startswith('{'):
                        try:
                            obj = json.loads(line.strip())
                            if 'title' in obj or 'description' in obj:
                                line_cases.append(obj)
                        except:
                            pass
                if line_cases:
                    test_cases = line_cases
                    break
    
    if test_cases:
        st.success(f"✅ LLM generated {len(test_cases)} test cases")
    else:
        st.warning(f"⚠️  LLM failed to generate test cases. Using fallback.")
    
    # Assign priority based on CVSS (if available) or OWASP severity
    if test_cases:
        highest_cvss = cves_sorted[0].get("cvss", 0) if cves_sorted else 0
        highest_cve = cves_sorted[0].get("cve_id", "") if cves_sorted else ""
        highest_cwe = cves_sorted[0].get("cwe", "") if cves_sorted else ""  # Extract from highest CVSS CVE
        
        # Map OWASP severity to estimated CVSS if no real CVSS available
        severity_to_cvss = {
            "Critical": 9.0,
            "High": 7.5,
            "Medium": 5.0,
            "Low": 3.0,
        }
        
        default_cvss = severity_to_cvss.get(owasp_vuln.get("severity", "Medium"), 5.0)
        final_cvss = highest_cvss if highest_cvss > 0 else default_cvss
        
        for tc in test_cases:
            # Assign unique ID using counter (will be reassigned after sorting)
            tc["id"] = test_case_counter
            test_case_counter += 1
            tc["owasp"] = owasp_vuln["owasp"]
            tc["priority"] = calculate_priority(final_cvss)
            tc["cvss_score"] = final_cvss  # Always include CVSS: real or estimated
            tc["cve_reference"] = highest_cve if highest_cvss > 0 else None
            tc["_has_real_cvss"] = bool(highest_cvss > 0)  # Explicit boolean
            if "metric" not in tc:
                tc["metric"] = {}
            tc["metric"]["cvss"] = final_cvss
            tc["metric"]["cwe"] = highest_cwe  # Store CWE from highest CVSS CVE
            tc["metric"]["cvss_source"] = "CVE" if highest_cvss > 0 else "OWASP_severity"
    
    return test_cases, test_case_counter



def generate_fallback_test_case(owasp_vuln: Dict, cves: List[Dict], test_id: int) -> Dict:
    """
    Generate a basic test case when LLM fails.
    
    Args:
        owasp_vuln: OWASP vulnerability definition
        cves: List of related CVEs
        test_id: REQUIRED - unique test case ID (no default to prevent ID collisions)
    
    Returns:
        Single test case with unique ID
    """
    # Use OWASP severity to estimate CVSS if no real CVE CVSS available
    severity_to_cvss = {
        "Critical": 9.0,
        "High": 7.5,
        "Medium": 5.0,
        "Low": 3.0,
    }
    cvss = severity_to_cvss.get(owasp_vuln.get("severity", "Medium"), 5.0)
    has_real_cvss = cves and cves[0].get("cvss", 0) > 0
    
    return {
        "id": test_id,  # Use passed ID - REQUIRED parameter, no default
        "title": f"Verify {owasp_vuln['name']} mitigation",
        "description": owasp_vuln["description"],
        "steps": [
            f"Set up test environment for {owasp_vuln['affected_component']}",
            f"Attempt to exploit {owasp_vuln['name'].lower()}",
            "Monitor for security violations or unauthorized actions",
            "Verify that preventive controls block the attack",
        ],
        "expected_result": (
            f"The system should either prevent the {owasp_vuln['name']} attack "
            "or detect and log it appropriately. No unauthorized access should be granted."
        ),
        "owasp": owasp_vuln["owasp"],
        "priority": calculate_priority(cvss),
        "cvss_score": cvss,  # Always include CVSS: real or estimated
        "cve_reference": cves[0]["cve_id"] if has_real_cvss else None,
        "_has_real_cvss": bool(has_real_cvss),  # Explicit boolean
        "metric": {
            "cvss": cvss, 
            "cwe": _extract_cwe_from_record(cves[0]) if cves else "",  # Use CWE extraction for fallback
            "cvss_source": "CVE" if has_real_cvss else "OWASP_severity",
        },
    }


def validate_test_case_ids(test_cases: List[Dict]) -> Tuple[bool, str]:
    """
    Validate that test case IDs are unique and sequential (1, 2, 3, ..., N).
    
    Returns:
        (is_valid, message) tuple
    """
    if not test_cases:
        return True, "No test cases to validate"
    
    ids = [tc.get("id") for tc in test_cases]
    unique_ids = set(ids)
    
    # Check for duplicates
    if len(unique_ids) != len(ids):
        duplicates = [id for id in unique_ids if ids.count(id) > 1]
        return False, f"Duplicate test case IDs detected: {duplicates}"
    
    # Check for sequential numbering
    expected_ids = set(range(1, len(test_cases) + 1))
    if unique_ids != expected_ids:
        missing = expected_ids - unique_ids
        unexpected = unique_ids - expected_ids
        return False, f"ID sequence broken. Missing: {missing}, Unexpected: {unexpected}"
    
    return True, f"✅ All {len(test_cases)} test cases have unique sequential IDs (1-{len(test_cases)})"


# =============================================================================
# UI & MAIN FLOW
# =============================================================================
def main():
    st.title("🔒 SecureTest AI v2")
    st.subheader("OWASP Vulnerability → Prioritized Test Cases")
    
    # Sidebar status + diagnostics
    with st.sidebar:
        st.header("System Status")
        ollama_ok, ollama_msg = check_ollama_ready()
        if ollama_ok:
            st.success(ollama_msg)
        else:
            st.error(f"❌ {ollama_msg}")
            return
        
        # Diagnostic info
        st.markdown("---")
        st.subheader("Diagnostics")
        st.info(
            f"**Model**: {OLLAMA_MODEL}\n"
            f"**Base URL**: {OLLAMA_BASE_URL}\n"
            f"**Temperature**: Varies by stage (0.6 OWASP, 0.7 Test Cases)\n"
            f"**Timeout**: 180 seconds\n\n"
            f"If results are identical every run, the LLM is likely failing JSON parsing or timing out. "
            f"Check console for error messages."
        )
    
    # Main input
    st.markdown("### 📝 Feature Description")
    feature_input = st.text_area(
        "Describe the feature you want to test:",
        placeholder="E.g., 'User authentication with OAuth2 and JWT tokens for mobile and web clients'",
        height=100,
    )
    
    if not feature_input.strip():
        st.info("Enter a feature description to begin")
        return
    
    # Generate button
    if st.button("🚀 Generate Security Tests", use_container_width=True):
        with st.spinner("Analyzing feature..."):
            # Load RAG
            collections = load_rag_collections()
            
            # Step 1: Identify OWASP vulnerabilities
            st.write("### Step 1️⃣: Identifying OWASP Vulnerabilities...")
            vulnerabilities = identify_owasp_vulnerabilities(feature_input)
            
            if not vulnerabilities:
                st.error("Could not identify vulnerabilities")
                return
            
            st.success(f"Found {len(vulnerabilities)} OWASP vulnerabilities")
            
            # Check if using hardcoded fallback
            hardcoded_owasp = {"A01:2021", "A03:2021", "A07:2021"}
            found_owasp = {v["owasp"] for v in vulnerabilities}
            if found_owasp == hardcoded_owasp:
                st.error(
                    "🚨 **WARNING**: Using hardcoded OWASP fallback! "
                    "This means the LLM failed to parse the response. "
                    "Check Ollama connection or try a different model."
                )
            
            # Step 2: Retrieve CVEs and patterns for each vulnerability
            st.write("### Step 2️⃣: Researching CVEs & Patterns...")
            all_test_cases = []
            test_id_counter = 1
            
            for vuln in vulnerabilities:
                with st.expander(f"🔍 {vuln['owasp']} — {vuln['name']}", expanded=False):
                    # Search CVEs
                    cves = search_cves_for_vulnerability(vuln, collections)
                    patterns = search_patterns_for_vulnerability(vuln, collections)
                    
                    st.write(f"**Severity:** {vuln['severity']}")
                    st.write(f"**Component:** {vuln['affected_component']}")
                    
                    if cves:
                        st.write(f"**Related CVEs:** {len(cves)}")
                        for cve in cves[:3]:
                            st.write(
                                f"  - {cve['cve_id']}: CVSS {cve['cvss']} "
                                f"({cve.get('severity', '?')})"
                            )
                    
                    if patterns:
                        st.write(f"**Mitigation Patterns:** {len(patterns)}")
                    
                    # Generate test cases
                    st.write("**Generating test cases...**")
                    test_cases, test_id_counter = generate_test_cases_for_vulnerability(
                        feature_input, vuln, cves, patterns, test_id_counter
                    )
                    
                    if not test_cases:
                        fallback_tc = generate_fallback_test_case(vuln, cves, test_id_counter)
                        test_cases = [fallback_tc]
                        test_id_counter += 1
                    
                    all_test_cases.extend(test_cases)
                    
                    st.write(f"✅ Generated {len(test_cases)} test cases")
            
            # Step 3: Display prioritized results
            st.write("### Step 3️⃣: Prioritized Test Results")
            
            # Sort by priority and CVSS (prioritize tests with real CVSS data)
            priority_order = {"P1": 1, "P2": 2, "P3": 3, "P4": 4}
            all_test_cases.sort(
                key=lambda x: (
                    priority_order.get(x.get("priority", "P4"), 5),
                    -(x.get("cvss_score", 0) or 0),
                    not x.get("_has_real_cvss", False),  # Real CVSS first
                )
            )
            
            # CRITICAL: Reassign IDs after sorting to maintain sequential numbering (1, 2, 3, ..., N)
            for idx, tc in enumerate(all_test_cases, 1):
                tc["id"] = idx
            
            # Re-validate IDs after reassignment
            id_valid, id_msg = validate_test_case_ids(all_test_cases)
            if id_valid:
                st.success(id_msg)
            else:
                st.error(f"Critical: {id_msg}")
            
            # Summary metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Total Test Cases", len(all_test_cases))
            with col2:
                p1_count = sum(1 for tc in all_test_cases if tc.get("priority") == "P1")
                st.metric("Critical (P1)", p1_count, delta=None)
            with col3:
                p2_count = sum(1 for tc in all_test_cases if tc.get("priority") == "P2")
                st.metric("High (P2)", p2_count, delta=None)
            with col4:
                max_cvss = max((tc.get("cvss_score", 0) or 0 for tc in all_test_cases), default=0)
                st.metric("Max CVSS", f"{max_cvss:.1f}")
            with col5:
                real_cvss_count = sum(1 for tc in all_test_cases if tc.get("_has_real_cvss"))
                st.metric("Real CVSS Data", f"{real_cvss_count}/{len(all_test_cases)}")
            
            st.markdown("---")
            
            # Display each test case
            for idx, tc in enumerate(all_test_cases, 1):
                priority = tc.get("priority", "P4")
                cvss = tc.get("cvss_score", 0)
                
                # Header with priority badge
                col1, col2, col3 = st.columns([1, 3, 2])
                with col1:
                    badge_class = f"badge-{priority.lower()}"
                    st.markdown(f"<div class='metric-badge {badge_class}'>{priority}</div>", unsafe_allow_html=True)
                with col2:
                    st.subheader(tc.get("title", f"Test Case {idx}"))
                with col3:
                    if cvss:
                        st.write(f"**CVSS:** {cvss}")
                    if tc.get("cve_reference"):
                        st.write(f"**CVE:** {tc['cve_reference']}")
                
                # Details
                st.write(f"**OWASP:** {tc.get('owasp', 'N/A')}")
                st.write(f"**Description:** {tc.get('description', 'N/A')}")
                cvss_source = tc.get("metric", {}).get("cvss_source", "unknown")
                if cvss_source == "OWASP_severity":
                    st.warning("⚠️  CVSS estimated from OWASP severity (no CVE with real CVSS score found)")
                
                # Steps
                st.write("**Test Steps:**")
                steps = tc.get("steps", [])
                if isinstance(steps, list):
                    for step_idx, step in enumerate(steps, 1):
                        st.write(f"  {step_idx}. {step}")
                else:
                    st.write(steps)
                
                # Expected result
                st.write(f"**Expected Result:** {tc.get('expected_result', 'N/A')}")
                
                # Metrics
                if tc.get("metric"):
                    metric_cols = st.columns(3)
                    with metric_cols[0]:
                        st.write(f"**CWE:** {tc['metric'].get('cwe', 'N/A')}")
                
                st.markdown("---")
            
            # Export option
            st.write("### 📥 Export Results")
            
            # Summary: Check if results are diverse or falling back to hardcoded
            fallback_count = sum(1 for tc in all_test_cases if tc.get("title", "").startswith("Verify "))
            is_mostly_fallback = fallback_count > len(all_test_cases) * 0.7
            
            if is_mostly_fallback:
                st.error(
                    "🚨 **WARNING**: Most test cases appear to be auto-generated fallbacks. "
                    "This indicates the LLM is failing to generate custom content.\n\n"
                    "**Troubleshooting**:\n"
                    "1. Check Ollama is running: `ollama serve`\n"
                    "2. Verify model exists: `ollama list`\n"
                    "3. Test LLM directly: `ollama run mistral \"hello\"`\n"
                    "4. Try a different model: Set OLLAMA_MODEL env var"
                )
            else:
                st.success("✅ LLM actively generating unique test cases (not relying on hardcoded fallbacks)")
            
            # FINAL CHECK: Verify IDs are correct before export
            final_ids = [tc.get("id") for tc in all_test_cases]
            if final_ids != list(range(1, len(all_test_cases) + 1)):
                st.error(f"⚠️ IDs incorrect before export: {final_ids}. Correcting...")
                for idx, tc in enumerate(all_test_cases, 1):
                    tc["id"] = idx
            
            export_json = json.dumps(all_test_cases, indent=2)
            st.download_button(
                label="Download as JSON",
                data=export_json,
                file_name=f"security_tests_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
            )


if __name__ == "__main__":
    main()
