"""
scenario1_case_upload.py
========================
Scenario 1: Case Upload & Instant Analysis
Downloads PDFs from S3, extracts with Groq, saves to output/judgments.json
"""

import boto3
import os
import json
import re
import time
import tempfile
import platform
import subprocess
from pathlib import Path
import PyPDF2
from groq import Groq
from dotenv import load_dotenv
from datetime import datetime
from botocore.config import Config
from botocore import UNSIGNED
from typing import List, Dict
# from dotenv import load_dotenv

class CaseUploadProcessor:
    """
    Process legal cases: Download PDF → Extract → Analyze → Save
    
    Meets Assignment Requirements:
    1. Extract metadata (court, date, acts, judges)
    2. Predict outcome (Allowed/Dismissed/Partly Allowed)
    3. Generate summary (<200 words)
    4. Save to output/judgments.json
    """
    
    def __init__(self, groq_api_key: str):
        self.s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
        self.bucket = "indian-high-court-judgments"
        self.groq = Groq(api_key=groq_api_key)
        self.model = "meta-llama/llama-4-scout-17b-16e-instruct"
        
        # Create output directory
        os.makedirs("output", exist_ok=True)
        
        # Track all processed cases
        self.all_cases = []
    
    def extract_bench_from_path(self, json_key: str) -> str:
        """Extract bench name from JSON metadata path"""
        bench_match = re.search(r'bench=([^/]+)', json_key)
        return bench_match.group(1) if bench_match else None
    
    def construct_pdf_path(self, json_key: str, year: str, court_code: str) -> str:
        """Construct correct S3 PDF path"""
        bench = self.extract_bench_from_path(json_key)
        if not bench:
            return None
        
        filename = os.path.basename(json_key).replace('.json', '.pdf')
        return f"data/pdf/year={year}/court={court_code}/bench={bench}/{filename}"
    
    def download_and_extract_pdf(self, pdf_s3_path: str) -> str:
        """Download PDF and extract text (Windows-safe)"""
        tmp_path = None
        
        try:
            # Create temp file
            tmp_file = tempfile.NamedTemporaryFile(mode='wb', suffix='.pdf', delete=False)
            tmp_path = tmp_file.name
            tmp_file.close()
            
            # Download PDF
            self.s3.download_file(self.bucket, pdf_s3_path, tmp_path)
            
            # Extract text
            text_parts = []
            with open(tmp_path, 'rb') as pdf_file:
                reader = PyPDF2.PdfReader(pdf_file)
                for page in reader.pages:
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text_parts.append(page_text)
                    except:
                        continue
            
            return "\n\n".join(text_parts)
        
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            return ""
        
        finally:
            # Cleanup
            if tmp_path and os.path.exists(tmp_path):
                try:
                    time.sleep(0.1)
                    os.unlink(tmp_path)
                except:
                    pass
    
    def analyze_case_with_groq(self, pdf_text: str, s3_path: str) -> Dict:
        """
        Analyze case with Groq LLM
        
        Returns structured JSON matching assignment requirements:
        - case_id
        - court, date
        - petitioners, respondents, judges
        - acts_referred
        - predicted_outcome (Allowed/Dismissed/Partly Allowed)
        - summary (<200 words)
        - source
        """
        
        if len(pdf_text) < 100:
            return None
        
        # Truncate to fit token limits
        max_chars = 15000
        if len(pdf_text) > max_chars:
            pdf_text = pdf_text[:max_chars]
        
        # Prompt optimized for assignment requirements
        prompt = f"""Extract structured information from this Indian High Court judgment and return ONLY valid JSON.

REQUIRED JSON FORMAT (exact schema):
{{
  "case_id": "COURT_HC_YEAR_NUMBER (e.g., MADRAS_HC_2021_02345)",
  "court": "Full court name (e.g., Madras High Court, Delhi High Court)",
  "date": "YYYY-MM-DD format (e.g., 2021-05-14)",
  "petitioners": ["List of petitioner names"],
  "respondents": ["List of respondent names"],
  "judges": ["List of judge names with designation"],
  "acts_referred": ["Acts and sections (e.g., IPC 302, CrPC 197)"],
  "predicted_outcome": "MUST BE ONE OF: Allowed, Dismissed, Partly Allowed, Disposed",
  "summary": "Concise summary under 200 words covering: what petitioner sought, court's reasoning, final decision"
}}

CRITICAL RULES:
1. predicted_outcome MUST be exactly: "Allowed" OR "Dismissed" OR "Partly Allowed" OR "Disposed"
2. summary MUST be under 200 words
3. Return ONLY the JSON object, no markdown, no extra text
4. Use empty arrays [] for missing lists
5. case_id format: COURTABBR_HC_YEAR_NUMBER

JUDGMENT TEXT:
{pdf_text}

JSON OUTPUT:"""
        
        # Call Groq with retry
        max_retries = 2
        for attempt in range(max_retries):
            try:
                response = self.groq.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a legal data extraction expert. Return ONLY valid JSON matching the exact schema provided. No markdown, no explanations."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0,
                    max_tokens=1500,  # Enough for summary <200 words
                    response_format={"type": "json_object"}
                )
                
                content = response.choices[0].message.content
                
                if not content:
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
                    return None
                
                # Clean and parse
                content = content.strip()
                # Remove leading and trailing Markdown code fences (e.g., ``` or ```json)
                content = re.sub(r'^```(?:\w+)?\s*', '', content)
                content = re.sub(r'\s*```$', '', content)
                
                extracted_data = json.loads(content)
                
                # Add source field (required by assignment)
                extracted_data["source"] = f"s3://{self.bucket}/{s3_path}"
                
                # Validate outcome
                valid_outcomes = ["Allowed", "Dismissed", "Partly Allowed", "Disposed"]
                if extracted_data.get("predicted_outcome") not in valid_outcomes:
                    # Try to fix common variations
                    outcome = extracted_data.get("predicted_outcome", "").lower()
                    if "allow" in outcome:
                        extracted_data["predicted_outcome"] = "Allowed"
                    elif "dismiss" in outcome:
                        extracted_data["predicted_outcome"] = "Dismissed"
                    elif "partly" in outcome or "partial" in outcome:
                        extracted_data["predicted_outcome"] = "Partly Allowed"
                    else:
                        extracted_data["predicted_outcome"] = "Disposed"
                
                # Validate summary length (should be <200 words)
                summary = extracted_data.get("summary", "")
                word_count = len(summary.split())
                if word_count > 200:
                    # Truncate to ~200 words
                    words = summary.split()[:200]
                    extracted_data["summary"] = " ".join(words) + "..."
                
                return extracted_data
            
            except json.JSONDecodeError as je:
                print(f"   ⚠️ JSON parse error: {str(je)}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return None
            
            except Exception as e:
                print(f"   ⚠️ Error: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return None
        
        return None
    
    def process_single_case(self, json_key: str, year: str, court_code: str) -> Dict:
        """Process a single case - complete pipeline"""
        
        case_id = os.path.basename(json_key).replace('.json', '')
        print(f"\n{'='*70}")
        print(f"📄 Processing: {case_id}")
        print(f"{'='*70}")
        
        # Construct PDF path
        pdf_path = self.construct_pdf_path(json_key, year, court_code)
        
        if not pdf_path:
            print(f"   ❌ Could not construct PDF path")
            return None
        
        print(f"   🔍 PDF path: {pdf_path}")
        
        # Verify PDF exists
        try:
            self.s3.head_object(Bucket=self.bucket, Key=pdf_path)
            print(f"   ✅ PDF found")
        except:
            print(f"   ❌ PDF not found in S3")
            return None
        
        # Download and extract text
        print(f"   📥 Downloading and extracting...")
        pdf_text = self.download_and_extract_pdf(pdf_path)
        
        if not pdf_text or len(pdf_text) < 100:
            print(f"   ⚠️ Insufficient text ({len(pdf_text)} chars)")
            return None
        
        print(f"   ✅ Extracted {len(pdf_text)} characters")
        
        # Analyze with Groq
        print(f"   🤖 Analyzing with Groq LLM...")
        case_data = self.analyze_case_with_groq(pdf_text, pdf_path)
        
        if not case_data:
            print(f"   ❌ Analysis failed")
            return None
        
        print(f"   ✅ Analysis complete")
        print(f"   📊 Outcome: {case_data.get('predicted_outcome')}")
        print(f"   📝 Summary: {case_data.get('summary', '')[:100]}...")
        
        return case_data
    
    def process_court(self, court_code: str, court_name: str, year: str = "2023", max_files: int = 50):
        """
        Process multiple cases from a court
        
        Args:
            court_code: Court code (e.g., "19_16" for Calcutta)
            court_name: Court display name
            year: Year to process
            max_files: Maximum number of cases to process
        """
        
        print(f"\n{'='*70}")
        print(f"🏛️  {court_name} - Year {year}")
        print(f"{'='*70}")
        
        # List JSON metadata files
        metadata_prefix = f"metadata/json/year={year}/court={court_code}/"
        
        try:
            response = self.s3.list_objects_v2(
                Bucket=self.bucket,
                Prefix=metadata_prefix,
                MaxKeys=max_files * 2
            )
            
            json_files = [
                obj["Key"] for obj in response.get("Contents", [])
                if obj["Key"].endswith(".json")
            ][:max_files]
            
            print(f"📂 Found {len(json_files)} JSON metadata files")
            
            processed_count = 0
            
            for json_key in json_files:
                case_data = self.process_single_case(json_key, year, court_code)
                
                if case_data:
                    self.all_cases.append(case_data)
                    processed_count += 1
                
                # Rate limiting
                time.sleep(2)
                
                # Stop if we have enough
                if processed_count >= max_files:
                    break
            
            print(f"\n✅ Processed {processed_count} cases from {court_name}")
        
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def save_to_output(self, filename: str = "judgments.json"):
        """
        Save all processed cases to output/judgments.json
        
        This matches the assignment requirement:
        "Persist structured JSON output to /output/judgments.json"
        """
        
        output_path = f"output/{filename}"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.all_cases, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*70}")
        print(f"💾 SAVED TO {output_path}")
        print(f"{'='*70}")
        print(f"Total cases: {len(self.all_cases)}")
        print(f"\nSample case:")
        if self.all_cases:
            print(json.dumps(self.all_cases[0], indent=2))
    
    def print_summary(self):
        """Print processing summary"""
        
        if not self.all_cases:
            print("\n⚠️ No cases processed")
            return
        
        # Count outcomes
        outcomes = {}
        for case in self.all_cases:
            outcome = case.get("predicted_outcome", "Unknown")
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
        
        print(f"\n{'='*70}")
        print(f"📊 PROCESSING SUMMARY")
        print(f"{'='*70}")
        print(f"Total cases processed: {len(self.all_cases)}")
        print(f"\nOutcome Distribution:")
        for outcome, count in outcomes.items():
            print(f"  {outcome}: {count}")
        print(f"{'='*70}")


# =============================================================================
# MAIN EXECUTION - Scenario 1
# =============================================================================

def run_scenario_1():
    """
    Scenario 1: Case Upload & Instant Analysis
    
    Requirements:
    ✓ Load file (PDF from S3)
    ✓ Extract metadata (court, date, acts, judges)
    ✓ Predict outcome (Allowed/Dismissed/Partly Allowed)
    ✓ Generate summary (<200 words)
    ✓ Save to output/judgments.json
    """
    
    print("\n" + "="*70)
    print("📋 SCENARIO 1: CASE UPLOAD & INSTANT ANALYSIS")
    print("="*70)
    
    # Check API key
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("\n❌ Error: GROQ_API_KEY not set")
        print("\nSet it with:")
        print("  PowerShell: $env:GROQ_API_KEY = \"gsk_<REDACTED_PRIMARY_KEY>\"")
        print("  CMD: set GROQ_API_KEY=gsk_your_key_here")
        return
    
    # Initialize processor
    processor = CaseUploadProcessor(groq_api_key=groq_api_key)
    
    # Process courts (as per assignment: Madras & Delhi)
    courts = [
        ("19_16", "Calcutta High Court"),  # Using Calcutta since PDFs exist
        # Can add more: ("33_10", "Madras High Court"), ("7_26", "Delhi High Court")
    ]
    
    for court_code, court_name in courts:
        processor.process_court(
            court_code=court_code,
            court_name=court_name,
            year="2023",
            max_files=10  # Process 10 cases per court for demo
        )
    
    # Save to output/judgments.json (assignment requirement)
    processor.save_to_output("judgments.json")
    
    # Print summary
    processor.print_summary()
    
    print("\n✅ SCENARIO 1 COMPLETE")
    print(f"📁 Output saved to: output/judgments.json")


def setup_api_key():
    """Set up GROQ API key with robust environment handling"""
    
    # Try loading from .env first
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        print(f"📄 Found .env file at: {env_path}")
        load_dotenv(env_path)
        
    # Check environment variable
    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        print(f"✅ GROQ_API_KEY loaded successfully")
        return api_key
        
    # On Windows, try reading from system environment
    if platform.system() == "Windows":
        try:
            # Try PowerShell system environment
            ps_cmd = "powershell -Command \"[System.Environment]::GetEnvironmentVariable('GROQ_API_KEY','User')\""
            result = subprocess.check_output(ps_cmd, shell=True, text=True).strip()
            if result:
                os.environ["GROQ_API_KEY"] = result
                print(f"✅ GROQ_API_KEY loaded from Windows system environment")
                return result
        except:
            pass
            
    # API key not found - print help
    print("\n❌ Error: GROQ_API_KEY not set")
    print("\nSet it with:")
    print("  PowerShell: $env:GROQ_API_KEY = \"gsk_<REDACTED_PRIMARY_KEY>\"")
    print("  CMD: set GROQ_API_KEY=gsk_your_key_here")
    return None


if __name__ == "__main__":
    print("\n" + "="*70)
    print("📋 SCENARIO 1: CASE UPLOAD & INSTANT ANALYSIS")
    print("="*70)
    
    # Set up API key first
    api_key = setup_api_key()
    if not api_key:
        exit(1)
    run_scenario_1()
