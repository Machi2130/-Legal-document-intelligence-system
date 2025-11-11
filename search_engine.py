"""
src/search_engine.py
====================
Search and filter legal cases
"""

import json
from typing import List, Dict, Optional

class CaseSearchEngine:
    """
    Search and filter legal judgments
    """
    
    def __init__(self, judgments_file: str = "output/judgments.json"):
        self.judgments_file = judgments_file
        self.cases = []
        self.load_cases()
    
    def load_cases(self):
        """Load cases from JSON"""
        try:
            with open(self.judgments_file, 'r', encoding='utf-8') as f:
                self.cases = json.load(f)
            print(f"✅ Loaded {len(self.cases)} cases")
        except FileNotFoundError:
            print(f"⚠️  No judgments file found")
            self.cases = []
    
    def search(self, 
               court: Optional[str] = None,
               act: Optional[str] = None,
               outcome: Optional[str] = None,
               keyword: Optional[str] = None) -> List[Dict]:
        """
        Search cases by various criteria
        """
        
        results = self.cases
        
        # Filter by court
        if court:
            results = [c for c in results if court.lower() in c.get('court', '').lower()]
        
        # Filter by act
        if act:
            results = [c for c in results 
                      if any(act.lower() in a.lower() for a in c.get('acts_referred', []))]
        
        # Filter by outcome
        if outcome:
            results = [c for c in results if c.get('predicted_outcome', '').lower() == outcome.lower()]
        
        # Filter by keyword
        if keyword:
            results = [c for c in results 
                      if keyword.lower() in c.get('summary', '').lower()]
        
        return results
    
    def get_statistics(self) -> Dict:
        """Get statistics about cases"""
        
        stats = {
            'total': len(self.cases),
            'by_court': {},
            'by_outcome': {}
        }
        
        for case in self.cases:
            # Count by court
            court = case.get('court', 'Unknown')
            stats['by_court'][court] = stats['by_court'].get(court, 0) + 1
            
            # Count by outcome
            outcome = case.get('predicted_outcome', 'Unknown')
            stats['by_outcome'][outcome] = stats['by_outcome'].get(outcome, 0) + 1
        
        return stats


if __name__ == "__main__":
    # Test
    engine = CaseSearchEngine()
    print(f"Search engine ready with {len(engine.cases)} cases")
