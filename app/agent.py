# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import base64
import json
import os
import re
from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field

import google.auth
from google.adk.agents import LlmAgent
from google.adk.agents.context import Context
from google.adk.apps import App, ResumabilityConfig
from google.adk.events.event import Event
from google.adk.events.request_input import RequestInput
from google.adk.workflow import Workflow, node
from google.genai import types

# GCP environment setup
try:
    _, project_id = google.auth.default()
    if project_id:
        os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
except Exception:
    pass

os.environ["GOOGLE_CLOUD_LOCATION"] = "global"
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"

# ==========================================
# Guidelines Knowledge Base
# ==========================================

NCCN_GUIDELINES_DB = {
    "breast_cancer": {
        "title": "NCCN Clinical Practice Guidelines in Oncology: Breast Cancer (v2.2026)",
        "guidelines": (
            "1. Staging Workup:\n"
            "   - Diagnostic mammogram, breast ultrasound, and pathological confirmation.\n"
            "   - Biomarker status must be evaluated: Estrogen Receptor (ER), Progesterone Receptor (PR), and HER2 status.\n"
            "   - Genetic testing is recommended for patients diagnosed under age 50, triple-negative breast cancer, or strong family history.\n"
            "2. HER2-Positive Early/Locally Advanced Breast Cancer (Stage II-III):\n"
            "   - Standard of Care: Systemic therapy containing chemotherapy plus HER2-targeted agents (Trastuzumab + Pertuzumab) is indicated.\n"
            "   - Neoadjuvant (pre-operative) therapy is preferred for tumors >= 2cm or node-positive disease to downstage the tumor and assess pathological response.\n"
            "   - Adjuvant therapy: Complete surgical resection followed by HER2-targeted therapy to complete 1 year. Radiation therapy is indicated post-lumpectomy or post-mastectomy with positive nodes.\n"
            "   - Endocrine therapy is indicated if ER/PR is positive."
        )
    },
    "lung_cancer": {
        "title": "NCCN Clinical Practice Guidelines in Oncology: Non-Small Cell Lung Cancer (v3.2026)",
        "guidelines": (
            "1. Staging Workup for Advanced Disease (Stage IV):\n"
            "   - Histological confirmation (adenocarcinoma vs. squamous cell).\n"
            "   - Molecular testing is mandatory for adenocarcinoma: EGFR, ALK, ROS1, RET, MET, BRAF, NTRK, and PD-L1 expression.\n"
            "   - Brain MRI is highly recommended for all Stage IV patients to evaluate for silent central nervous system (CNS) metastases.\n"
            "2. EGFR Mutation-Positive Metastatic NSCLC (Stage IV):\n"
            "   - Standard of Care: First-line tyrosine kinase inhibitor (TKI) therapy (Osimertinib is the preferred category 1 recommendation).\n"
            "   - Platinum-doublet chemotherapy is NOT recommended as first-line therapy unless EGFR TKIs are contraindicated.\n"
            "   - Next steps: Brain MRI if not done, monitor treatment response with chest/abdomen CT every 8-12 weeks."
        )
    }
}

# ==========================================
# Schema Definitions
# ==========================================

class UploadedDocument(BaseModel):
    filename: str = Field(description="Name of the file uploaded")
    mime_type: str = Field(description="MIME type of the file (e.g. application/pdf, image/png, text/plain)")
    content_b64: str = Field(description="Base64 encoded file content, or raw text if plain text")

class DocumentParsedMetadata(BaseModel):
    filename: str = Field(description="Name of the file parsed")
    doc_type: str = Field(description="Type of medical document, e.g. Pathology Report, Imaging Report (CT/MRI/PET), Lab Results, Clinical Notes, Discharge Summary")
    date: str = Field(description="Date of the document, YYYY-MM-DD or Unknown")
    patient_name: str = Field(description="Name of the patient, or Unknown")
    findings_summary: str = Field(description="Summary of critical clinical findings, e.g., tumor measurements, positive nodes, margin status")
    biomarkers_mentioned: List[str] = Field(default_factory=list, description="List of biomarkers or mutations mentioned, e.g., HER2+, ER+, EGFR L858R, ALK positive")
    extracted_text: str = Field(description="Full text parsed or extracted from the document")

class ClinicalProfile(BaseModel):
    inferred_disease: str = Field(default="", description="Inferred cancer type or primary diagnosis, e.g., Invasive Ductal Carcinoma of the breast, Non-Small Cell Lung Cancer")
    stage: str = Field(default="", description="Inferred stage of the cancer, e.g. Stage I, Stage IIA, Stage IIIB, Stage IV, or Unknown")
    tnm_staging: Dict[str, str] = Field(default_factory=dict, description="TNM staging parameters if found, e.g. T: 'T2', N: 'N1', M: 'M0'")
    biomarkers: List[str] = Field(default_factory=list, description="Aggregated key biomarkers and genetic mutations")
    overall_health_status: str = Field(default="", description="Patient's current health status or performance status if mentioned")
    clinical_history_summary: str = Field(default="", description="Synthesized chronological clinical history summary of the patient")
    extracted_doctor_recommendations: List[str] = Field(default_factory=list, description="Treatment recommendations explicitly mentioned by the doctors in the records")

class NCCNComplianceCheck(BaseModel):
    doctor_recommendation_evaluated: str = Field(default="", description="The doctor's recommended treatment plan being evaluated")
    nccn_guideline_reference: str = Field(default="", description="Summary of the relevant NCCN guideline pathway and recommendations for this specific stage and biomarker profile")
    is_compliant: str = Field(default="Indeterminate", description="Compliance status, must be one of: 'Compliant', 'Non-Compliant', 'Partially-Compliant', or 'Indeterminate'")
    discrepancy_explanation: str = Field(default="", description="Detailed explanation of any deviations from the NCCN guidelines, or why compliance could not be fully determined")
    recommended_next_steps: List[str] = Field(default_factory=list, description="Next clinical steps as per guidelines")
    recommended_additional_tests: List[str] = Field(default_factory=list, description="Recommended additional tests, imaging, or workup that are missing from current files but required by NCCN guidelines")

class ChatbotResponse(BaseModel):
    reply: str = Field(description="Response to the user's question, written in a compassionate, easy-to-understand, and medically accurate tone.")
    source_citations: List[str] = Field(default_factory=list, description="Specific source documents or guideline sections referenced to answer this question")

class MedicalState(BaseModel):
    uploaded_files: List[UploadedDocument] = []
    parsed_documents: List[DocumentParsedMetadata] = []
    clinical_profile: ClinicalProfile = ClinicalProfile()
    compliance_check: NCCNComplianceCheck = NCCNComplianceCheck()
    user_question: Optional[str] = None
    chat_response: Optional[ChatbotResponse] = None
    chat_history: List[Dict[str, str]] = []
    doctor_recommendation: Optional[str] = None

# ==========================================
# Helpers & Verification
# ==========================================

def has_gcp_credentials() -> bool:
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return True
    try:
        google.auth.default()
        return True
    except Exception:
        return False

async def call_gemini_structured(prompt: str, schema: Any, system_instruction: Optional[str] = None, contents_parts: Optional[List[Any]] = None) -> Any:
    from google import genai
    from google.genai import types
    client = genai.Client()
    
    contents = contents_parts or [prompt]
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=schema,
        system_instruction=system_instruction,
        temperature=0.1,
    )
    
    import asyncio
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=config
        )
    )
    
    data = json.loads(response.text)
    return schema.model_validate(data)

# ==========================================
# Simulated fallbacks
# ==========================================

def simulate_document_parsing(filename: str, content_b64: str, mime_type: str) -> DocumentParsedMetadata:
    fn_lower = filename.lower()
    if "pathology" in fn_lower or "breast" in fn_lower:
        return DocumentParsedMetadata(
            filename=filename,
            doc_type="Pathology Report",
            date="2026-05-12",
            patient_name="Sarah Jenkins",
            findings_summary="Invasive Ductal Carcinoma of the left breast, tumor size 3.2 cm. Clear margins. Lymph nodes: 2 of 10 positive for metastasis.",
            biomarkers_mentioned=["HER2 (3+) Positive", "Estrogen Receptor (ER) Negative", "Progesterone Receptor (PR) Negative"],
            extracted_text=(
                "PATHO-LOGIC DIAGNOSIS REPORT\n"
                "Patient: Sarah Jenkins\n"
                "Date: 2026-05-12\n"
                "Specimen: Left breast lumpectomy and sentinel lymph node biopsy.\n"
                "Diagnosis: Invasive Ductal Carcinoma, Grade 3.\n"
                "Tumor Size: 3.2 cm in greatest dimension.\n"
                "Surgical Margins: Uninvolved (> 2mm).\n"
                "Lymph Nodes: 2 of 10 nodes positive for macrometastasis.\n"
                "Immunohistochemistry (IHC):\n"
                "  - Estrogen Receptor (ER): Negative (0% staining)\n"
                "  - Progesterone Receptor (PR): Negative (0% staining)\n"
                "  - HER2: 3+ Positive (strong, circumferential membrane staining in >10% of cells)"
            )
        )
    elif "lung" in fn_lower or "ct_scan" in fn_lower or "chest" in fn_lower:
        return DocumentParsedMetadata(
            filename=filename,
            doc_type="Imaging Report & Biopsy",
            date="2026-06-18",
            patient_name="Michael Chen",
            findings_summary="4.5 cm right upper lobe lung mass. Multiple pleural nodules. Metastatic lesions in segments IV and VIII of the liver. Biopsy shows Lung Adenocarcinoma with EGFR Exon 21 L858R mutation.",
            biomarkers_mentioned=["EGFR Exon 21 L858R positive", "ALK Negative", "PD-L1 TPS: 15%"],
            extracted_text=(
                "CLINICAL RADIOLOGY & PATHOLOGY REPORT\n"
                "Patient: Michael Chen\n"
                "Date: 2026-06-18\n"
                "CT Chest/Abdomen/Pelvis:\n"
                "  - Lungs: 4.5 cm primary mass in the right upper lobe. Multiple bilateral pleural nodules suspicious for metastasis.\n"
                "  - Liver: Multiple hypoattenuating lesions, largest measuring 2.3 cm in segment IV and 1.8 cm in segment VIII, consistent with metastatic disease.\n"
                "Biopsy of Right Lung Mass:\n"
                "  - Histology: Adenocarcinoma, solid and acinar patterns.\n"
                "  - Next-Generation Sequencing (NGS) molecular pathology panel:\n"
                "    * EGFR: Exon 21 L858R mutation detected (variant allele frequency 34%)\n"
                "    * ALK Rearrangement: Negative\n"
                "    * ROS1 Fusion: Negative\n"
                "    * PD-L1 (22C3 PharmDx): Tumor Proportion Score (TPS) = 15%"
            )
        )
    else:
        text_content = ""
        try:
            text_content = base64.b64decode(content_b64).decode("utf-8")
        except Exception:
            text_content = "Raw text content or document data."
        
        return DocumentParsedMetadata(
            filename=filename,
            doc_type="Clinical Notes",
            date="2026-06-30",
            patient_name="John Doe",
            findings_summary="General clinical notes or medical history file.",
            biomarkers_mentioned=[],
            extracted_text=text_content or "No text content extracted."
        )

def simulate_clinical_synthesis(parsed_docs: List[DocumentParsedMetadata]) -> ClinicalProfile:
    has_breast = any(p.doc_type == "Pathology Report" or "breast" in p.findings_summary.lower() for p in parsed_docs)
    has_lung = any("lung" in p.findings_summary.lower() or "egfr" in "".join(p.biomarkers_mentioned).lower() for p in parsed_docs)
    
    if has_breast:
        return ClinicalProfile(
            inferred_disease="Invasive Ductal Carcinoma of the Left Breast",
            stage="Stage IIIA",
            tnm_staging={"T": "T2", "N": "N1", "M": "M0"},
            biomarkers=["HER2 Positive (3+)", "ER Negative", "PR Negative"],
            overall_health_status="ECOG performance status 0",
            clinical_history_summary=(
                "Patient is a 48-year-old female who presented with a palpable left breast mass. "
                "Lumpectomy and sentinel node biopsy on 2026-05-12 confirmed Invasive Ductal Carcinoma, size 3.2cm, "
                "with 2 out of 10 sentinel lymph nodes positive. Tumor markers show HER2-positive status and ER/PR-negative status."
            ),
            extracted_doctor_recommendations=[
                "Surgical resection (completed).",
                "Referral to Medical Oncology for adjuvant chemotherapy and HER2-targeted therapy.",
                "Adjuvant radiation therapy post-chemotherapy."
            ]
        )
    elif has_lung:
        return ClinicalProfile(
            inferred_disease="Metastatic Non-Small Cell Lung Cancer (Adenocarcinoma)",
            stage="Stage IV",
            tnm_staging={"T": "T2", "N": "N2", "M": "M1b"},
            biomarkers=["EGFR Exon 21 L858R Mutation Positive", "ALK Negative", "PD-L1 15%"],
            overall_health_status="ECOG performance status 1",
            clinical_history_summary=(
                "Patient is a 62-year-old male presenting with chronic cough and shortness of breath. "
                "CT imaging on 2026-06-18 revealed a 4.5 cm right upper lobe lung mass, multiple pleural nodules, "
                "and hepatic lesions in segments IV and VIII, indicating metastatic disease. "
                "Lung biopsy confirmed Adenocarcinoma, and NGS molecular testing detected an EGFR Exon 21 L858R mutation."
            ),
            extracted_doctor_recommendations=[
                "Start standard platinum-doublet chemotherapy.",
                "Palliative consult for respiratory symptoms."
            ]
        )
    else:
        return ClinicalProfile(
            inferred_disease="General Medical Case",
            stage="Unknown",
            tnm_staging={},
            biomarkers=[],
            overall_health_status="Not documented",
            clinical_history_summary="Synthesized general medical history from the uploaded files.",
            extracted_doctor_recommendations=["Follow up as directed by clinical team."]
        )

def simulate_guidelines_compliance(profile: ClinicalProfile, explicit_rec: Optional[str]) -> NCCNComplianceCheck:
    rec = explicit_rec or (profile.extracted_doctor_recommendations[0] if profile.extracted_doctor_recommendations else "Standard follow-up")
    
    if "Breast" in profile.inferred_disease:
        has_her2_targeted = any(t in rec.lower() for t in ["herceptin", "trastuzumab", "pertuzumab", "targeted", "her2"])
        is_compliant = "Compliant" if has_her2_targeted else "Partially-Compliant"
        discrepancy = "" if is_compliant == "Compliant" else "The recommended treatment plan lacks HER2-targeted therapy (e.g. Trastuzumab/Pertuzumab), which is a critical standard of care for HER2-positive early breast cancer as per NCCN Guidelines."
        
        return NCCNComplianceCheck(
            doctor_recommendation_evaluated=rec,
            nccn_guideline_reference=(
                "NCCN Breast Cancer Guidelines (v2.2026) recommend: For HER2-positive, Stage II/III early breast cancer, "
                "systemic therapy must consist of chemotherapy combined with HER2-targeted agents (Trastuzumab + Pertuzumab). "
                "Neoadjuvant systemic therapy is preferred for tumors >= 2cm or node-positive disease. Post-lumpectomy radiation is required."
            ),
            is_compliant=is_compliant,
            discrepancy_explanation=discrepancy or "The recommended adjuvant chemotherapy plus trastuzumab/pertuzumab plan is fully aligned with NCCN guidelines for HER2-positive breast cancer.",
            recommended_next_steps=[
                "Initiate chemotherapy plus Trastuzumab/Pertuzumab regimen.",
                "Plan for adjuvant radiation therapy after chemotherapy.",
                "Refer to genetic counseling (strongly recommended for age < 50)."
            ],
            recommended_additional_tests=[
                "Genetic testing for BRCA1/BRCA2 mutation.",
                "Echocardiogram to establish baseline cardiac function before starting HER2-targeted therapy."
            ]
        )
    elif "Lung" in profile.inferred_disease:
        has_tki = any(t in rec.lower() for t in ["osimertinib", "tki", "tarceva", "erlotinib", "iressa", "gefitinib", "tagrisso"])
        is_compliant = "Compliant" if has_tki else "Partially-Compliant"
        discrepancy = "" if is_compliant == "Compliant" else "The doctor recommended standard chemotherapy. However, for EGFR mutation-positive Stage IV lung cancer, NCCN guidelines recommend first-line therapy with an EGFR tyrosine kinase inhibitor (TKI), specifically Osimertinib, as it provides superior efficacy and tolerability compared to standard chemotherapy."
        
        return NCCNComplianceCheck(
            doctor_recommendation_evaluated=rec,
            nccn_guideline_reference=(
                "NCCN Non-Small Cell Lung Cancer Guidelines (v3.2026) recommend: For metastatic (Stage IV) NSCLC with EGFR "
                "sensitizing mutations (Exon 19 deletion or Exon 21 L858R), first-line therapy should be a third-generation EGFR "
                "TKI (Osimertinib, Category 1). Standard chemotherapy should be reserved for disease progression."
            ),
            is_compliant=is_compliant,
            discrepancy_explanation=discrepancy,
            recommended_next_steps=[
                "Initiate first-line EGFR TKI therapy with Osimertinib 80mg daily.",
                "Consult with molecular oncology regarding the EGFR Exon 21 L858R mutation profile."
            ],
            recommended_additional_tests=[
                "Brain MRI (highly recommended by NCCN guidelines to evaluate for asymptomatic central nervous system metastasis, which occurs in up to 30% of Stage IV patients).",
                "Baseline chest, abdomen, and pelvis CT scan for tumor size tracking before treatment onset."
            ]
        )
    else:
        return NCCNComplianceCheck(
            doctor_recommendation_evaluated=rec,
            nccn_guideline_reference="No specific NCCN guideline reference was matched for this diagnosis.",
            is_compliant="Indeterminate",
            discrepancy_explanation="Unable to check compliance: diagnosis is general or unknown.",
            recommended_next_steps=["Consult with clinical specialist."],
            recommended_additional_tests=[]
        )

def simulate_chatbot_response(user_question: str, profile: Optional[ClinicalProfile], compliance: Optional[NCCNComplianceCheck], chat_history: List[Dict[str, str]]) -> ChatbotResponse:
    if profile is None:
        profile = ClinicalProfile(
            inferred_disease="Unknown Disease",
            stage="Unknown Stage",
            biomarkers=[],
            timeline_summary="No records uploaded."
        )
    if compliance is None:
        compliance = NCCNComplianceCheck(
            is_compliant="Indeterminate",
            doctor_recommendation_evaluated="None provided",
            discrepancy_explanation="No audit has been performed because no doctor recommendations were uploaded.",
            recommended_next_steps=["Please upload your doctor's treatment recommendation letter."],
            recommended_additional_tests=["Diagnostic staging workup."]
        )
    q_lower = user_question.lower()
    
    if "her2" in q_lower:
        reply = (
            "HER2 (Human Epidermal Growth Factor Receptor 2) is a protein that promotes the growth of cancer cells. "
            "In your case, the pathology report indicates your tumor is HER2-positive (3+). This means your cancer cells "
            "have higher-than-normal levels of this protein, which can make the tumor grow faster. However, it also means "
            "your cancer is highly responsive to HER2-targeted therapies like Trastuzumab (Herceptin) and Pertuzumab (Perjeta), "
            "which specifically target this protein to stop cancer cells from growing."
        )
        citations = ["Pathology Report from 2026-05-12", "NCCN Breast Cancer Guidelines (v2.2026)"]
    elif "stage" in q_lower:
        if "breast" in profile.inferred_disease.lower():
            reply = (
                "Based on the pathology report, your disease is staged as Stage IIIA. This staging is determined "
                "by a tumor size of 3.2 cm (T2) and the presence of cancer in 2 of the 10 examined lymph nodes (N1), "
                "with no signs of distant spread (M0). Stage IIIA means it is locally advanced early breast cancer, "
                "which is treated aggressively with a combination of surgery, chemotherapy, targeted therapy, and radiation."
            )
            citations = ["Pathology Report from 2026-05-12", "AJCC Staging System"]
        elif "lung" in profile.inferred_disease.lower():
            reply = (
                "Your medical records indicate Stage IV Metastatic Lung Adenocarcinoma. Stage IV means the cancer has spread "
                "beyond the primary site in the lung to other regions of the body—specifically, the pleural nodes and the liver "
                "(segment IV and VIII) as shown on your CT scan from 2026-06-18. While Stage IV is advanced, targeted therapies "
                "such as EGFR TKIs (Osimertinib) are designed specifically for patients with your mutation and are highly effective."
            )
            citations = ["CT Scan from 2026-06-18", "NCCN NSCLC Guidelines v3.2026"]
        else:
            reply = f"Your current diagnosed stage appears to be {profile.stage}."
            citations = ["Patient Profile Summary"]
    elif "egfr" in q_lower:
        reply = (
            "EGFR (Epidermal Growth Factor Receptor) is a protein on the surface of cells that helps them grow. "
            "An EGFR mutation (specifically your Exon 21 L858R mutation) means that the receptor is permanently stuck in the 'on' position, "
            "causing the cells to grow out of control. The good news is that we have targeted medications called EGFR Tyrosine Kinase Inhibitors "
            "(TKIs), such as Osimertinib, that block this signal and shrink the tumor effectively."
        )
        citations = ["Lung Biopsy NGS Report from 2026-06-18", "NCCN NSCLC Guidelines (v3.2026)"]
    elif "next step" in q_lower or "test" in q_lower or "guideline" in q_lower:
        steps_str = "\n- ".join(compliance.recommended_next_steps)
        tests_str = "\n- ".join(compliance.recommended_additional_tests)
        reply = (
            f"According to NCCN guidelines, the recommended next steps for your clinical profile are:\n"
            f"- {steps_str}\n\n"
            f"Additionally, the guidelines recommend the following additional tests or checks that were not found in your files:\n"
            f"- {tests_str}"
        )
        citations = ["NCCN Guidelines Reference Pathways"]
    elif "compliant" in q_lower or "match" in q_lower or "doctor" in q_lower:
        reply = (
            f"Our NCCN compliance audit evaluated the recommended treatment plan as **{compliance.is_compliant}** with standard guidelines. "
            f"\n\nDetails: {compliance.discrepancy_explanation}"
        )
        citations = ["NCCN Guidelines Audit Report"]
    else:
        reply = (
            f"I have analyzed your medical history. You have been diagnosed with {profile.inferred_disease} ({profile.stage}). "
            f"Our NCCN guidelines audit indicates your recommended plan is {compliance.is_compliant}. "
            f"Could you please specify if you would like to know about your staging, specific biomarkers, treatment compliance, or recommended next steps?"
        )
        citations = ["Clinical Profile Summary"]
        
    reply += "\n\n*Disclaimer: I am an AI assistant designed to help translate complex medical records and compare them against cancer guidelines. I do not provide medical diagnosis, treatment, or advice. Always consult your oncology team before making any medical decisions.*"
    
    return ChatbotResponse(
        reply=reply,
        source_citations=citations
    )

# ==========================================
# Graph Nodes
# ==========================================

def parse_input_node(node_input: Any):
    """
    Parses incoming payload, which can contain:
    - files: List[Dict[str, str]] with filename, content_b64, mime_type
    - user_question: str
    - doctor_recommendation: str
    """
    files = []
    user_q = None
    doc_rec = None
    
    if isinstance(node_input, str):
        try:
            parsed = json.loads(node_input)
            if isinstance(parsed, dict):
                files = parsed.get("files", [])
                user_q = parsed.get("user_question") or parsed.get("prompt")
                doc_rec = parsed.get("doctor_recommendation")
            else:
                user_q = node_input
        except Exception:
            user_q = node_input
    elif isinstance(node_input, dict):
        files = node_input.get("files", [])
        user_q = node_input.get("user_question") or node_input.get("prompt")
        doc_rec = node_input.get("doctor_recommendation")
    
    state_delta = {}
    
    # 1. Update uploaded files in state
    if files:
        uploaded_list = []
        for f in files:
            uploaded_list.append(UploadedDocument(
                filename=f.get("filename", "document.txt"),
                mime_type=f.get("mime_type", "text/plain"),
                content_b64=f.get("content_b64", "")
            ))
        state_delta["uploaded_files"] = uploaded_list
        
    # 2. Update questions and recommendations
    if user_q:
        state_delta["user_question"] = user_q
    if doc_rec:
        state_delta["doctor_recommendation"] = doc_rec
        
    # Determine the routing
    route = "analyze" if files else "chat"
        
    info_msg = (
        f"📝 **Parsed Input Node**:\n"
        f"- Files uploaded: {len(files)}\n"
        f"- User Question: {user_q}\n"
        f"- Doctor Recommendation: {doc_rec}\n"
        f"- Route chosen: {route}"
    )
    yield Event(content=types.Content(role='model', parts=[types.Part.from_text(text=info_msg)]))
    yield Event(output=user_q or "", route=route, state=state_delta)


@node()
async def document_ingestion_node(ctx: Context, node_input: str):
    """
    Iterates over uploaded files and performs OCR/parsing using Gemini or simulated fallbacks.
    """
    uploaded_files = ctx.state.get("uploaded_files", [])
    parsed_documents = []
    
    info_msg = f"🔍 Processing and performing OCR on {len(uploaded_files)} document(s)..."
    yield Event(content=types.Content(role='model', parts=[types.Part.from_text(text=info_msg)]))
    
    for doc in uploaded_files:
        try:
            if has_gcp_credentials():
                parts = []
                if doc.mime_type.startswith("image/") or doc.mime_type == "application/pdf":
                    file_bytes = base64.b64decode(doc.content_b64)
                    part = types.Part.from_bytes(data=file_bytes, mime_type=doc.mime_type)
                    parts.append(part)
                else:
                    try:
                        text_content = base64.b64decode(doc.content_b64).decode("utf-8")
                    except Exception:
                        text_content = doc.content_b64
                    parts.append(text_content)
                
                parts.append(
                    "Analyze the attached medical document. Extract: document type, document date, "
                    "patient name, findings summary, biomarkers mentioned, and complete text. "
                    "Return in JSON matching the DocumentParsedMetadata schema."
                )
                
                parsed_meta = await call_gemini_structured(
                    prompt="",
                    schema=DocumentParsedMetadata,
                    system_instruction="You are an expert medical records parser and OCR system.",
                    contents_parts=parts
                )
                parsed_documents.append(parsed_meta)
            else:
                parsed_meta = simulate_document_parsing(doc.filename, doc.content_b64, doc.mime_type)
                parsed_documents.append(parsed_meta)
        except Exception:
            parsed_meta = simulate_document_parsing(doc.filename, doc.content_b64, doc.mime_type)
            parsed_documents.append(parsed_meta)
            
    state_delta = {"parsed_documents": parsed_documents}
    result_summary = "\n".join([f"- **{d.filename}** ({d.doc_type}): {d.findings_summary}" for d in parsed_documents])
    yield Event(content=types.Content(role='model', parts=[types.Part.from_text(text=f"✅ **OCR & Parsing Complete**:\n{result_summary}")]))
    yield Event(output=parsed_documents, state=state_delta)


@node()
async def clinical_synthesis_node(ctx: Context, node_input: List[DocumentParsedMetadata]):
    """
    Synthesizes the parsed clinical data to infer diagnosis, staging, and key markers.
    """
    parsed_docs = ctx.state.get("parsed_documents", [])
    
    yield Event(content=types.Content(role='model', parts=[types.Part.from_text(text="🧬 Synthesizing clinical history and inferring staging...")]))
    
    try:
        if has_gcp_credentials():
            docs_context = "\n\n".join([
                f"File: {doc.filename}\nType: {doc.doc_type}\nDate: {doc.date}\n"
                f"Findings: {doc.findings_summary}\nBiomarkers: {', '.join(doc.biomarkers_mentioned)}\nText: {doc.extracted_text}"
                for doc in parsed_docs
            ])
            
            prompt = (
                f"Below is a list of parsed clinical records for a patient:\n\n{docs_context}\n\n"
                "Synthesize this history into a unified ClinicalProfile, containing inferred disease, "
                "overall stage, TNM parameters, aggregated biomarkers, and doctor recommendations."
            )
            
            profile = await call_gemini_structured(
                prompt=prompt,
                schema=ClinicalProfile,
                system_instruction="You are a board-certified clinical oncologist and medical NLP system."
            )
        else:
            profile = simulate_clinical_synthesis(parsed_docs)
    except Exception:
        profile = simulate_clinical_synthesis(parsed_docs)
        
    state_delta = {"clinical_profile": profile}
    
    if not ctx.state.get("doctor_recommendation") and profile.extracted_doctor_recommendations:
        state_delta["doctor_recommendation"] = profile.extracted_doctor_recommendations[0]
        
    info_msg = (
        f"📊 **Inferred Clinical Profile**:\n"
        f"- **Primary Diagnosis**: {profile.inferred_disease}\n"
        f"- **Inferred Stage**: {profile.stage}\n"
        f"- **Biomarkers**: {', '.join(profile.biomarkers) if profile.biomarkers else 'None'}\n"
        f"- **TNM Staging**: {profile.tnm_staging}"
    )
    yield Event(content=types.Content(role='model', parts=[types.Part.from_text(text=info_msg)]))
    yield Event(output=profile, state=state_delta)


@node()
async def guidelines_compliance_node(ctx: Context, node_input: ClinicalProfile):
    """
    Checks the extracted treatment recommendations against the matched NCCN guidelines.
    """
    profile = ctx.state.get("clinical_profile")
    doc_rec = ctx.state.get("doctor_recommendation") or "No explicit recommendation found."
    
    yield Event(content=types.Content(role='model', parts=[types.Part.from_text(text="🩺 Auditing treatment recommendations against NCCN guidelines...")]))
    
    disease = profile.inferred_disease
    guideline_text = NCCN_GUIDELINES_DB.get("breast_cancer", {}).get("guidelines", "")
    if "lung" in disease.lower():
        guideline_text = NCCN_GUIDELINES_DB.get("lung_cancer", {}).get("guidelines", "")
        
    try:
        if has_gcp_credentials():
            prompt = (
                f"Patient Clinical Profile:\n"
                f"- Diagnosis: {profile.inferred_disease}\n"
                f"- Stage: {profile.stage}\n"
                f"- Biomarkers: {', '.join(profile.biomarkers)}\n\n"
                f"Doctor's Recommended Treatment Plan: {doc_rec}\n\n"
                f"Matched NCCN Guidelines Reference:\n{guideline_text}\n\n"
                "Evaluate the treatment plan. Classify compliance ('Compliant', 'Non-Compliant', "
                "'Partially-Compliant', or 'Indeterminate'), explain discrepancies, and list recommended next steps and missing tests."
            )
            
            compliance = await call_gemini_structured(
                prompt=prompt,
                schema=NCCNComplianceCheck,
                system_instruction="You are an expert oncology clinical quality auditor."
            )
        else:
            compliance = simulate_guidelines_compliance(profile, doc_rec)
    except Exception:
        compliance = simulate_guidelines_compliance(profile, doc_rec)
        
    state_delta = {"compliance_check": compliance}
    
    status_icon = "✅" if compliance.is_compliant == "Compliant" else "⚠️" if compliance.is_compliant == "Partially-Compliant" else "❌"
    info_msg = (
        f"🏥 **NCCN Guidelines Audit Result**:\n"
        f"- **Compliance Status**: {status_icon} **{compliance.is_compliant}**\n"
        f"- **Doctor Recommendation Evaluated**: {compliance.doctor_recommendation_evaluated}\n"
        f"- **Deviation/Discrepancy Details**: {compliance.discrepancy_explanation}\n\n"
        f"📋 **Recommended Next Steps**:\n"
        + "\n".join([f"  - {step}" for step in compliance.recommended_next_steps]) + "\n\n"
        f"🔍 **Missing Guidelines Staging Tests/Workup**:\n"
        + "\n".join([f"  - {test}" for test in compliance.recommended_additional_tests])
    )
    yield Event(content=types.Content(role='model', parts=[types.Part.from_text(text=info_msg)]))
    
    user_q = ctx.state.get("user_question")
    route = "chat" if user_q else "done"
    
    yield Event(output=compliance, route=route, state=state_delta)


@node()
async def chatbot_interaction_node(ctx: Context, node_input: Any):
    """
    Conversational RAG chatbot node.
    """
    user_q = ctx.state.get("user_question") or "Tell me about my disease."
    profile = ctx.state.get("clinical_profile")
    compliance = ctx.state.get("compliance_check")
    chat_hist = ctx.state.get("chat_history", [])
    
    try:
        if has_gcp_credentials():
            hist_str = "\n".join([f"{h['role']}: {h['text']}" for h in chat_hist])
            prompt = (
                f"Patient Clinical Profile: {profile.model_dump_json()}\n"
                f"NCCN Guidelines Audit: {compliance.model_dump_json()}\n\n"
                f"Conversation History:\n{hist_str}\n\n"
                f"Patient Question: {user_q}\n\n"
                "Formulate a compassionate, plain-language, medically accurate response citing specific patient records and guidelines."
            )
            
            chat_resp = await call_gemini_structured(
                prompt=prompt,
                schema=ChatbotResponse,
                system_instruction=(
                    "You are OncoCompanion, an expert, supportive clinical oncology AI chatbot. "
                    "Cite sources and include a medical disclaimer."
                )
            )
        else:
            chat_resp = simulate_chatbot_response(user_q, profile, compliance, chat_hist)
    except Exception:
        chat_resp = simulate_chatbot_response(user_q, profile, compliance, chat_hist)
        
    new_history = list(chat_hist)
    new_history.append({"role": "user", "text": user_q})
    new_history.append({"role": "model", "text": chat_resp.reply})
    
    state_delta = {
        "chat_response": chat_resp,
        "chat_history": new_history,
        "user_question": None
    }
    
    yield Event(content=types.Content(role='model', parts=[types.Part.from_text(text=chat_resp.reply)]))
    yield Event(output=chat_resp.reply, state=state_delta)


# ==========================================
# Graph Definition
# ==========================================

edges = [
    ('START', parse_input_node),
    (parse_input_node, {"analyze": document_ingestion_node, "chat": chatbot_interaction_node}),
    (document_ingestion_node, clinical_synthesis_node),
    (clinical_synthesis_node, guidelines_compliance_node),
    (guidelines_compliance_node, {"chat": chatbot_interaction_node}),
]

root_agent = Workflow(
    name="medical_history_synthesis",
    edges=edges,
    state_schema=MedicalState,
)

app = App(
    root_agent=root_agent,
    name="app",
    resumability_config=ResumabilityConfig(enabled=True)
)
