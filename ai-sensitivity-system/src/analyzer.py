import os
import json
import yaml
from pathlib import Path
from datetime import datetime
from collections import defaultdict

try:
    import pandas as pd
    import openpyxl
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

BASE_DIR = Path(__file__).parent.parent

class SensitivityAnalyzer:
    def __init__(self, config_path=None):
        if config_path is None:
            config_path = BASE_DIR / "config.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        
        def _resolve_path(p):
            p = Path(p)
            return p if p.is_absolute() else BASE_DIR / p
        
        self.processed_dir = _resolve_path(self.config["data"]["processed_dir"])
        self.outputs_dir = _resolve_path(self.config["data"]["outputs_dir"])
        self.confidence_threshold = self.config["threshold"]["confidence"]
        
        self.outputs_dir.mkdir(exist_ok=True)
    
    def _load_results(self):
        results = []
        for file_path in self.processed_dir.glob("*.json"):
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                results.extend(data)
        return results
    
    def analyze(self):
        results = self._load_results()
        
        if not results:
            print("No results found to analyze")
            return None
        
        stats = {
            "total_records": len(results),
            "analyzed_at": datetime.now().isoformat(),
            "keyword_stats": {},
            "pattern_distribution": defaultdict(int),
            "overall_metrics": {
                "related_count": 0,
                "not_related_count": 0,
                "low_confidence_count": 0,
                "false_positive_proxy": 0
            }
        }
        
        keyword_results = defaultdict(list)
        pattern_counts = defaultdict(int)
        
        for result in results:
            ai_result = result.get("ai_result", {})
            related = ai_result.get("related", False)
            confidence = ai_result.get("confidence", 0)
            pattern_type = ai_result.get("pattern_type", "other")
            
            keyword = result.get("keyword", "unknown")
            keyword_results[keyword].append(ai_result)
            pattern_counts[pattern_type] += 1
            
            if related:
                stats["overall_metrics"]["related_count"] += 1
                if confidence < self.confidence_threshold:
                    stats["overall_metrics"]["false_positive_proxy"] += 1
            else:
                stats["overall_metrics"]["not_related_count"] += 1
            
            if confidence < self.confidence_threshold:
                stats["overall_metrics"]["low_confidence_count"] += 1
        
        stats["pattern_distribution"] = dict(pattern_counts)
        
        for keyword, keyword_result_list in keyword_results.items():
            kw_stats = {
                "total": len(keyword_result_list),
                "related_count": sum(1 for r in keyword_result_list if r.get("related", False)),
                "not_related_count": sum(1 for r in keyword_result_list if not r.get("related", False)),
                "avg_confidence": sum(r.get("confidence", 0) for r in keyword_result_list) / len(keyword_result_list),
                "low_confidence_count": sum(1 for r in keyword_result_list if r.get("confidence", 0) < self.confidence_threshold),
                "pattern_distribution": defaultdict(int)
            }
            
            for r in keyword_result_list:
                pattern = r.get("pattern_type", "other")
                kw_stats["pattern_distribution"][pattern] += 1
            
            kw_stats["pattern_distribution"] = dict(kw_stats["pattern_distribution"])
            kw_stats["false_positive_proxy_rate"] = kw_stats["low_confidence_count"] / kw_stats["total"] if kw_stats["total"] > 0 else 0
            
            stats["keyword_stats"][keyword] = kw_stats
        
        stats["overall_metrics"]["false_positive_proxy_rate"] = (
            stats["overall_metrics"]["false_positive_proxy"] / stats["overall_metrics"]["related_count"] 
            if stats["overall_metrics"]["related_count"] > 0 else 0
        )
        
        stats_file = self.outputs_dir / "stats.json"
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        if HAS_PANDAS:
            self._generate_excel_report(stats)
        
        print(f"Analysis complete. Stats saved to {stats_file}")
        return stats
    
    def _generate_excel_report(self, stats):
        with pd.ExcelWriter(self.outputs_dir / "report.xlsx", engine="openpyxl") as writer:
            overview_df = pd.DataFrame({
                "Metric": [
                    "Total Records",
                    "Related Count",
                    "Not Related Count",
                    "Low Confidence Count",
                    "False Positive Proxy Rate"
                ],
                "Value": [
                    stats["overall_metrics"]["total_records"],
                    stats["overall_metrics"]["related_count"],
                    stats["overall_metrics"]["not_related_count"],
                    stats["overall_metrics"]["low_confidence_count"],
                    f"{stats['overall_metrics']['false_positive_proxy_rate']:.2%}"
                ]
            })
            overview_df.to_excel(writer, sheet_name="Overview", index=False)
            
            keyword_data = []
            for keyword, kw_stats in stats["keyword_stats"].items():
                keyword_data.append({
                    "Keyword": keyword,
                    "Total": kw_stats["total"],
                    "Related": kw_stats["related_count"],
                    "Not Related": kw_stats["not_related_count"],
                    "Avg Confidence": f"{kw_stats['avg_confidence']:.4f}",
                    "Low Confidence Count": kw_stats["low_confidence_count"],
                    "False Positive Proxy Rate": f"{kw_stats['false_positive_proxy_rate']:.2%}"
                })
            keywords_df = pd.DataFrame(keyword_data)
            keywords_df.to_excel(writer, sheet_name="Keyword Stats", index=False)
            
            pattern_data = []
            for pattern, count in stats["pattern_distribution"].items():
                percentage = count / stats["overall_metrics"]["total_records"] * 100
                pattern_data.append({
                    "Pattern Type": pattern,
                    "Count": count,
                    "Percentage": f"{percentage:.2f}%"
                })
            patterns_df = pd.DataFrame(pattern_data)
            patterns_df.to_excel(writer, sheet_name="Pattern Distribution", index=False)

if __name__ == "__main__":
    analyzer = SensitivityAnalyzer()
    analyzer.analyze()
