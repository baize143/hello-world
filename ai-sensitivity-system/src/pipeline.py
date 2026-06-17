import os
import json
import csv
import yaml
from pathlib import Path
from datetime import datetime

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from judge import LLMSemanticJudge

BASE_DIR = Path(__file__).parent.parent

class SensitivityPipeline:
    def __init__(self, config_path=None):
        if config_path is None:
            config_path = BASE_DIR / "config.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        
        def _resolve_path(p):
            p = Path(p)
            return p if p.is_absolute() else BASE_DIR / p
        
        self.raw_dir = _resolve_path(self.config["data"]["raw_dir"])
        self.processed_dir = _resolve_path(self.config["data"]["processed_dir"])
        self.cases_dir = _resolve_path(self.config["data"]["cases_dir"])
        self.confidence_threshold = self.config["threshold"]["confidence"]
        self.prompt_version = self.config["prompt"]["version"]
        self.judge = LLMSemanticJudge(config_path)
        
        self.processed_dir.mkdir(exist_ok=True)
        self.cases_dir.mkdir(exist_ok=True)
    
    def _read_csv(self, file_path):
        if HAS_PANDAS:
            return pd.read_csv(file_path)
        else:
            rows = []
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append(row)
            return rows
    
    def _read_excel(self, file_path):
        if HAS_PANDAS:
            return pd.read_excel(file_path)
        else:
            raise ValueError("pandas required to read Excel files")
    
    def _read_file(self, file_path):
        ext = file_path.suffix.lower()
        if ext == ".csv":
            return self._read_csv(file_path)
        elif ext in [".xlsx", ".xls"]:
            return self._read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    
    def _mark_low_confidence(self, result):
        if result.get("confidence", 0) < self.confidence_threshold:
            result["low_confidence"] = True
        else:
            result["low_confidence"] = False
        return result
    
    def _save_misjudged(self, text, keyword, ai_result):
        misjudged_path = self.cases_dir / "misjudged.json"
        if misjudged_path.exists():
            with open(misjudged_path, "r", encoding="utf-8") as f:
                cases = json.load(f)
        else:
            cases = []
        
        case = {
            "text": text,
            "keyword": keyword,
            "ai_result": ai_result,
            "human_label": "",
            "timestamp": datetime.now().isoformat(),
            "version": f"judge_{self.prompt_version}"
        }
        cases.append(case)
        
        with open(misjudged_path, "w", encoding="utf-8") as f:
            json.dump(cases, f, ensure_ascii=False, indent=2)
    
    def run(self):
        results = []
        files = list(self.raw_dir.glob("*"))
        
        for file_path in files:
            if file_path.is_file():
                try:
                    data = self._read_file(file_path)
                    if HAS_PANDAS:
                        records = data.to_dict("records")
                    else:
                        records = data
                    
                    for record in records:
                        text = record.get("消息文本", record.get("text", ""))
                        keyword = record.get("命中敏感词", record.get("keyword", ""))
                        
                        if text and keyword:
                            ai_result = self.judge.judge(text, keyword)
                            ai_result = self._mark_low_confidence(ai_result)
                            
                            if ai_result.get("low_confidence", False):
                                self._save_misjudged(text, keyword, ai_result)
                            
                            result = {
                                "text": text,
                                "keyword": keyword,
                                "ai_result": ai_result,
                                "source_file": file_path.name,
                                "processed_at": datetime.now().isoformat()
                            }
                            results.append(result)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
        
        output_file = self.processed_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"Processed {len(results)} records")
        print(f"Results saved to {output_file}")
        return results

if __name__ == "__main__":
    pipeline = SensitivityPipeline()
    pipeline.run()
