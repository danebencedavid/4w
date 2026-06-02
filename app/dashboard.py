"""Static dashboard entry point.

Run from the project root:

    python app/dashboard.py
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from debrecen_weather.config import load_config  # noqa: E402
from debrecen_weather.reporting import write_html_report  # noqa: E402


if __name__ == "__main__":
    cfg = load_config(ROOT / "configs" / "debrecen.yml")
    report = write_html_report(cfg)
    print(report)
