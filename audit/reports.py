import json
from datetime import datetime

def generate_report(model_name, attack, metrics, period):
    report = {
        "model": model_name,
        "attack": attack,
        "metrics": metrics,
        "period": period,
        "generated_at": datetime.utcnow().isoformat()
    }

    with open(f"fraudstruct_audit_{period}.json", "w") as f:
        json.dump(report, f, indent=2)

    return report
