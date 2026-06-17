import requests
import json
import re
import yaml
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

class LLMSemanticJudge:
    def __init__(self, config_path=None):
        if config_path is None:
            config_path = BASE_DIR / "config.yaml"
        with open(config_path, "r", encoding="utf-8") as cf:
            self.config = yaml.safe_load(cf)
        self.model_name = self.config["model"]["name"]
        self.api_endpoint = self.config["model"]["api_endpoint"]
        self.confidence_threshold = self.config["threshold"]["confidence"]
        self.prompt_version = self.config["prompt"]["version"]
        self.prompt_base_path = Path(self.config["prompt"]["base_path"])
        if not self.prompt_base_path.is_absolute():
            self.prompt_base_path = BASE_DIR / self.prompt_base_path
        prompt_file = self.prompt_base_path / f"judge_{self.prompt_version}.txt"
        with open(prompt_file, "r", encoding="utf-8") as pf:
            self.prompt_template = pf.read()

    def _extract_json(self, text):
        pattern = r"\\{[\\s\\S]*\\}"
        matches = re.findall(pattern, text)
        return matches[-1] if matches else None

    def _safe_parse_json(self, text):
        try:
            json_str = self._extract_json(text)
            return json.loads(json_str) if json_str else None
        except:
            return None

    def judge(self, text, keyword):
        prompt = self.prompt_template.replace("{text}", text).replace("{keyword}", keyword)
        payload = {"model": self.model_name, "prompt": prompt, "stream": False, "temperature": 0.1}
        try:
            resp = requests.post(self.api_endpoint, json=payload, timeout=60)
            resp.raise_for_status()
            result = resp.json()
            if "response" in result:
                raw_output = result["response"]
                parsed = self._safe_parse_json(raw_output)
                if parsed and isinstance(parsed, dict):
                    parsed["raw_response"] = raw_output
                    parsed["timestamp"] = result.get("created_at", "")
                    return parsed
                return {"related": False, "confidence": 0.0, "pattern_type": "other", "reason": "Parse failed", "raw_response": raw_output}
            return {"related": False, "confidence": 0.0, "pattern_type": "other", "reason": "No response"}
        except requests.exceptions.RequestException as e:
            return {"related": False, "confidence": 0.0, "pattern_type": "other", "reason": f"Request failed: {e}"}
        except Exception as e:
            return {"related": False, "confidence": 0.0, "pattern_type": "other", "reason": f"Error: {e}"}

if __name__ == "__main__":
    judge = LLMSemanticJudge()
    result = judge.judge("test", "keyword")
    print(json.dumps(result, ensure_ascii=False, indent=2))
