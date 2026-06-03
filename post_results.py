"""Morning results script — runs daily at 10 AM ET via GitHub Actions."""
from datetime import date, timedelta

from dotenv import load_dotenv


def main() -> None:
    load_dotenv()
    yesterday = (date.today() - timedelta(days=1)).isoformat()

    from agents.outcome_tracker import update_for_date
    from agents.discord_bot import post_results, post_weekly_recap

    print(f"Updating outcomes for {yesterday}...")
    try:
        update_for_date(yesterday)
    except Exception as e:
        print(f"  [post_results] update_for_date failed: {e} — continuing to post results")

    print("Posting results to Discord...")
    post_results(yesterday)

    if date.today().weekday() == 6:  # Sunday
        print("Sunday — posting weekly recap...")
        post_weekly_recap()

    print("Done.")


if __name__ == "__main__":
    main()
