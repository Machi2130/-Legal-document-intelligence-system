"""
api.py
======
Flask REST API for Legal Intelligence System
Connects with your HTML frontend
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, 'src')

from fetch_data import DataFetcher
from entity_extractor import EntityExtractor
from preprocess import DataPreprocessor
from search_engine import CaseSearchEngine
from similarity_engine import SimilarityCaseFinder
from analytics import LegalAnalytics

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Global variables
JUDGMENTS_FILE = "output/judgments.json"

# Initialize components
def get_search_engine():
    """Get or create search engine instance"""
    if not os.path.exists(JUDGMENTS_FILE):
        return None
    return CaseSearchEngine(JUDGMENTS_FILE)

def get_similarity_engine():
    """Get or create similarity engine instance"""
    if not os.path.exists(JUDGMENTS_FILE):
        return None
    return SimilarityCaseFinder(JUDGMENTS_FILE)

def get_analytics_engine():
    """Get or create analytics engine instance"""
    if not os.path.exists(JUDGMENTS_FILE):
        return None
    return LegalAnalytics(JUDGMENTS_FILE)


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.route('/api/status', methods=['GET'])
def get_status():

    try:
        if not os.path.exists(JUDGMENTS_FILE):
            return jsonify({
                "status": "no_data",
                "total_cases": 0,
                "total_courts": 0,
                "total_judges": 0,
                "total_acts": 0,
                "message": "No data available. Please download data first."
            })
        
        # Load judgments
        with open(JUDGMENTS_FILE, 'r', encoding='utf-8') as f:
            cases = json.load(f)
        
        # Calculate statistics
        courts = set()
        judges = set()
        acts = set()
        
        for case in cases:
            courts.add(case.get('court', 'Unknown'))
            judges.update(case.get('judges', []))
            acts.update(case.get('acts_referred', []))
        
        return jsonify({
            "status": "ready",
            "total_cases": len(cases),
            "total_courts": len(courts),
            "total_judges": len(judges),
            "total_acts": len(acts)
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/search', methods=['POST'])
def search_cases():
   
    try:
        search_engine = get_search_engine()
        if not search_engine:
            return jsonify({"error": "No data available"}), 404
        
        data = request.get_json()
        
        # Extract search parameters
        query = data.get('query', '')
        court = data.get('court', '')
        outcome = data.get('outcome', '')
        case_type = data.get('case_type', '')
        
        # Perform search
        results = search_engine.search(
            court=court if court else None,
            outcome=outcome if outcome else None,
            keyword=query if query else None
        )
        
        # Filter by case type if provided
        if case_type:
            results = [
                c for c in results 
                if case_type.lower() in c.get('case_type', '').lower()
            ]
        
        return jsonify({
            "results": results[:50],  # Limit to 50 results
            "count": len(results)
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/similarity', methods=['POST'])
def find_similar_cases():
   
    try:
        similarity_engine = get_similarity_engine()
        if not similarity_engine:
            return jsonify({"error": "No data available"}), 404
        
        data = request.get_json()
        
        query_text = data.get('query_text', '')
        top_k = data.get('top_k', 5)
        
        if not query_text:
            return jsonify({"error": "query_text is required"}), 400
        
        # Find similar cases
        results = similarity_engine.find_similar_by_text(query_text, topk=top_k)
        
        # Format results
        formatted_results = [
            {
                "case": case,
                "similarity": float(similarity)
            }
            for case, similarity in results
        ]
        
        return jsonify({
            "results": formatted_results,
            "count": len(formatted_results)
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/analytics/<analysis_type>', methods=['GET'])
def get_analytics(analysis_type):
    
    try:
        analytics_engine = get_analytics_engine()
        if not analytics_engine:
            return jsonify({"error": "No data available"}), 404
        
        if analysis_type == 'judges':
            # Analyze judges
            judge_stats = {}
            for case in analytics_engine.cases:
                judges = case.get('judges', [])
                outcome = case.get('predicted_outcome', 'Unknown')
                
                for judge in judges:
                    if judge not in judge_stats:
                        judge_stats[judge] = {
                            'total_cases': 0,
                            'outcomes': {}
                        }
                    
                    judge_stats[judge]['total_cases'] += 1
                    judge_stats[judge]['outcomes'][outcome] = judge_stats[judge]['outcomes'].get(outcome, 0) + 1
            
            # Format and sort
            result = []
            for judge, stats in judge_stats.items():
                total = stats['total_cases']
                percentages = {
                    outcome: round((count / total) * 100, 1)
                    for outcome, count in stats['outcomes'].items()
                }
                
                result.append({
                    'judge': judge,
                    'total_cases': total,
                    'outcomes': stats['outcomes'],
                    'outcome_percentages': percentages
                })
            
            result.sort(key=lambda x: x['total_cases'], reverse=True)
            return jsonify({"data": result[:20]})  # Top 20
        
        elif analysis_type == 'acts':
            # Most cited acts
            act_counts = {}
            for case in analytics_engine.cases:
                acts = case.get('acts_referred', [])
                for act in acts:
                    act_counts[act] = act_counts.get(act, 0) + 1
            
            result = [
                {"act": act, "total_cases": count}
                for act, count in sorted(act_counts.items(), key=lambda x: x[1], reverse=True)
            ]
            
            return jsonify({"data": result[:20]})  # Top 20
        
        elif analysis_type == 'courts':
            # Court distribution
            court_stats = {}
            for case in analytics_engine.cases:
                court = case.get('court', 'Unknown')
                if court not in court_stats:
                    court_stats[court] = {
                        'total_cases': 0,
                        'outcomes': {}
                    }
                
                court_stats[court]['total_cases'] += 1
                outcome = case.get('predicted_outcome', 'Unknown')
                court_stats[court]['outcomes'][outcome] = court_stats[court]['outcomes'].get(outcome, 0) + 1
            
            return jsonify({"data": court_stats})
        
        elif analysis_type == 'outcomes':
            # Outcome distribution
            outcome_counts = {}
            for case in analytics_engine.cases:
                outcome = case.get('predicted_outcome', 'Unknown')
                outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
            
            return jsonify({"data": outcome_counts})
        
        else:
            return jsonify({"error": "Invalid analysis type"}), 400
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/download', methods=['POST'])
def download_and_process():
    
    try:
        # Check API key
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            return jsonify({"error": "GROQ_API_KEY not set"}), 500
        
        data = request.get_json()
        year = data.get('year', '2023')
        max_files = data.get('max_files', 20)
        
        # Step 1: Fetch PDFs
        fetcher = DataFetcher()
        
        delhi_data = fetcher.fetch_court_data('delhi', year=year, max_files=max_files)
        madras_data = fetcher.fetch_court_data('madras', year=year, max_files=max_files)
        
        all_fetched = delhi_data + madras_data
        files_downloaded = len(all_fetched)
        
        if files_downloaded == 0:
            return jsonify({"error": "No files downloaded"}), 404
        
        # Step 2: Extract with Groq
        extractor = EntityExtractor(groq_api_key=groq_api_key)
        extracted_cases = extractor.extract_batch(all_fetched)
        cases_processed = len(extracted_cases)
        
        # Step 3: Preprocess
        preprocessor = DataPreprocessor()
        processed_cases = preprocessor.process(extracted_cases)
        
        # Step 4: Merge with existing data
        existing_cases = []
        if os.path.exists(JUDGMENTS_FILE):
            with open(JUDGMENTS_FILE, 'r', encoding='utf-8') as f:
                existing_cases = json.load(f)
        
        # Remove duplicates
        existing_ids = {c.get('case_id') for c in existing_cases}
        new_cases = [c for c in processed_cases if c.get('case_id') not in existing_ids]
        
        # Combine
        all_cases = existing_cases + new_cases
        
        # Save
        os.makedirs("output", exist_ok=True)
        with open(JUDGMENTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(all_cases, f, indent=2, ensure_ascii=False)
        
        # Rebuild similarity index
        try:
            if os.path.exists("output/embeddings.pkl"):
                os.remove("output/embeddings.pkl")
            
            similarity_engine = SimilarityCaseFinder(JUDGMENTS_FILE)
        except:
            pass  # Index will be rebuilt on next similarity search
        
        return jsonify({
            "files_downloaded": files_downloaded,
            "cases_processed": cases_processed,
            "cases_added": len(new_cases),
            "total_cases": len(all_cases)
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "timestamp": datetime.now().isoformat()
    })


# =============================================================================
# RUN SERVER
# =============================================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 LEGAL INTELLIGENCE SYSTEM - API SERVER")
    print("="*70)
    print("\n📡 Starting Flask server on http://localhost:5000")
    print("\nAPI Endpoints:")
    print("  GET  /api/status         - System status")
    print("  POST /api/search         - Search cases")
    print("  POST /api/similarity     - Find similar cases")
    print("  GET  /api/analytics/<type> - Get analytics")
    print("  POST /api/download       - Download & process data")
    print("  GET  /api/health         - Health check")
    print("\n" + "="*70)
    print("\n⚠️  Make sure GROQ_API_KEY is set:")
    print(f"   GROQ_API_KEY: {'✅ Set' if os.getenv('GROQ_API_KEY') else '❌ Not Set'}")
    print("\n" + "="*70)
    
    app.run(host='0.0.0.0', port=5000, debug=True)
