"""Phase-2 CLV closing-line capture.

Idempotent. Run repeatedly during game hours (e.g. every 10 min, ~5-11 PM ET
via Task Scheduler). Each pass fills the closing Pinnacle line, CLV, and
confirmed-lineup flag for any logged play whose first pitch is within the next
30 minutes and has no closing line yet. Safe to run with no pending plays.
"""
import os
from datetime import datetime, timezone

from dotenv import load_dotenv

from agents.clv_log import capture_closing


def main() -> None:
    load_dotenv()
    api_key = os.environ["ODDS_API_KEY"]
    now = datetime.now(timezone.utc)
    capture_closing(api_key, now)
    print(f"Closing-capture pass complete at {now.isoformat()}.")


if __name__ == "__main__":
    main()
