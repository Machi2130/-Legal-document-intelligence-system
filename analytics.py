from collections import defaultdict
from datetime import datetime
from typing import Dict, List, DefaultDict
import json
import os
import matplotlib.pyplot as plt


class LegalAnalytics:
    
    def __init__(self, judgments_file: str = "output/judgments.json") -> None:
        self.judgments_file = judgments_file
        self.cases: List[Dict] = []
        self._load_cases()
    
    def _load_cases(self) -> None:
        try:
            with open(self.judgments_file, 'r', encoding='utf-8') as file:
                self.cases = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            self.cases = []
    
    def analyze_by_judge(self) -> Dict[str, Dict]:
        judge_stats: DefaultDict[str, Dict] = defaultdict(lambda: {
            "total_cases": 0,
            "outcomes": defaultdict(int),
            "courts": set(),
            "years": set()
        })
        
        for case in self.cases:
            judges = case.get("judges", [])
            outcome = case.get("predicted_outcome", "Unknown")
            court = case.get("court", "Unknown")
            year = self._extract_year(case.get("date", "Unknown"))
            
            for judge in judges:
                judge_stats[judge]["total_cases"] += 1
                judge_stats[judge]["outcomes"][outcome] += 1
                judge_stats[judge]["courts"].add(court)
                judge_stats[judge]["years"].add(year)
        
        return self._format_judge_statistics(judge_stats)
    
    def _extract_year(self, date: str) -> str:
        return date[:4] if date != "Unknown" and len(date) >= 4 else "Unknown"
    
    def _format_judge_statistics(self, judge_stats: DefaultDict[str, Dict]) -> Dict[str, Dict]:
        result = {}
        for judge, stats in judge_stats.items():
            result[judge] = {
                "total_cases": stats["total_cases"],
                "outcomes": dict(stats["outcomes"]),
                "outcome_percentages": self._calculate_percentages(stats["outcomes"]),
                "courts": list(stats["courts"]),
                "years_active": sorted([y for y in stats["years"] if y != "Unknown"])
            }
        return result
    
    def analyze_by_act(self) -> Dict[str, Dict]:
        act_stats: DefaultDict[str, Dict] = defaultdict(lambda: {
            "total_cases": 0,
            "outcomes": defaultdict(int),
            "courts": defaultdict(int)
        })
        
        for case in self.cases:
            acts = case.get("acts_referred", [])
            outcome = case.get("predicted_outcome", "Unknown")
            court = case.get("court", "Unknown")
            
            for act in acts:
                act_stats[act]["total_cases"] += 1
                act_stats[act]["outcomes"][outcome] += 1
                act_stats[act]["courts"][court] += 1
        
        return self._format_act_statistics(act_stats)
    
    def _format_act_statistics(self, act_stats: DefaultDict[str, Dict]) -> Dict[str, Dict]:
        result = {}
        for act, stats in act_stats.items():
            result[act] = {
                "total_cases": stats["total_cases"],
                "outcomes": dict(stats["outcomes"]),
                "outcome_percentages": self._calculate_percentages(stats["outcomes"]),
                "courts": dict(stats["courts"])
            }
        return result
    
    def analyze_by_court(self) -> Dict[str, Dict]:
        court_stats: DefaultDict[str, Dict] = defaultdict(lambda: {
            "total_cases": 0,
            "outcomes": defaultdict(int),
            "case_types": defaultdict(int),
            "years": defaultdict(int)
        })
        
        for case in self.cases:
            court = case.get("court", "Unknown")
            outcome = case.get("predicted_outcome", "Unknown")
            case_type = case.get("case_type", "Unknown")
            year = self._extract_year(case.get("date", "Unknown"))
            
            court_stats[court]["total_cases"] += 1
            court_stats[court]["outcomes"][outcome] += 1
            court_stats[court]["case_types"][case_type] += 1
            court_stats[court]["years"][year] += 1
        
        return self._format_court_statistics(court_stats)
    
    def _format_court_statistics(self, court_stats: DefaultDict[str, Dict]) -> Dict[str, Dict]:
        result = {}
        for court, stats in court_stats.items():
            result[court] = {
                "total_cases": stats["total_cases"],
                "outcomes": dict(stats["outcomes"]),
                "outcome_percentages": self._calculate_percentages(stats["outcomes"]),
                "case_types": dict(stats["case_types"]),
                "year_distribution": dict(sorted(stats["years"].items()))
            }
        return result
    
    def get_top_judges(self, top_n: int = 10, sort_by: str = "total_cases") -> List[Dict]:
        judge_stats = self.analyze_by_judge()
        
        for judge, stats in judge_stats.items():
            outcomes = stats["outcomes"]
            total = stats["total_cases"]
            stats["allowed_rate"] = outcomes.get("Allowed", 0) / total if total > 0 else 0
            stats["dismissed_rate"] = outcomes.get("Dismissed", 0) / total if total > 0 else 0
        
        sorted_judges = sorted(judge_stats.items(), key=lambda x: x[1].get(sort_by, 0), reverse=True)
        return [{"judge": judge, **stats} for judge, stats in sorted_judges[:top_n]]
    
    def get_top_acts(self, top_n: int = 10) -> List[Dict]:
        act_stats = self.analyze_by_act()
        sorted_acts = sorted(act_stats.items(), key=lambda x: x[1]["total_cases"], reverse=True)
        return [{"act": act, **stats} for act, stats in sorted_acts[:top_n]]
    
    def visualize_judge_outcomes(self, output_dir: str = "output/visualizations") -> None:
        os.makedirs(output_dir, exist_ok=True)
        top_judges = self.get_top_judges(top_n=10)
        
        if not top_judges:
            return
        
        judges = [j["judge"][:30] for j in top_judges]
        allowed = [j["outcomes"].get("Allowed", 0) for j in top_judges]
        dismissed = [j["outcomes"].get("Dismissed", 0) for j in top_judges]
        partly = [j["outcomes"].get("Partly Allowed", 0) for j in top_judges]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        x_positions = range(len(judges))
        bar_width = 0.6
        
        ax.bar(x_positions, allowed, bar_width, label='Allowed', color='#2ecc71')
        ax.bar(x_positions, dismissed, bar_width, bottom=allowed, label='Dismissed', color='#e74c3c')
        ax.bar(x_positions, partly, bar_width, bottom=[a + d for a, d in zip(allowed, dismissed)], label='Partly Allowed', color='#f39c12')
        
        ax.set_xlabel('Judges', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Cases', fontsize=12, fontweight='bold')
        ax.set_title('Case Outcomes by Top 10 Judges', fontsize=14, fontweight='bold')
        ax.set_xticks(x_positions)
        ax.set_xticklabels(judges, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "judge_outcomes.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def visualize_court_distribution(self, output_dir: str = "output/visualizations") -> None:
        os.makedirs(output_dir, exist_ok=True)
        court_stats = self.analyze_by_court()
        
        if not court_stats:
            return
        
        courts = list(court_stats.keys())
        counts = [court_stats[c]["total_cases"] for c in courts]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
        
        ax.pie(counts, labels=courts, autopct='%1.1f%%', colors=colors, startangle=90)
        ax.set_title('Case Distribution by Court', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "court_distribution.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self, output_file: str = "output/analytics_report.json") -> Dict:
        report = {
            "metadata": {
                "total_cases": len(self.cases),
                "report_generated": datetime.now().isoformat()
            },
            "by_judge": self.analyze_by_judge(),
            "by_act": self.analyze_by_act(),
            "by_court": self.analyze_by_court(),
            "top_judges": self.get_top_judges(10),
            "top_acts": self.get_top_acts(10)
        }
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as file:
            json.dump(report, file, indent=2, ensure_ascii=False)
        
        return report
    
    def _calculate_percentages(self, outcome_dict: Dict[str, int]) -> Dict[str, float]:
        total = sum(outcome_dict.values())
        if total == 0:
            return {}
        return {k: round(v / total * 100, 2) for k, v in outcome_dict.items()}


if __name__ == "__main__":
    analytics = LegalAnalytics()
    analytics.generate_report()
    analytics.visualize_judge_outcomes()
    analytics.visualize_court_distribution()