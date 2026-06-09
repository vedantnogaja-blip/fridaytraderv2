import schedule
import time
import subprocess
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

def run_agent():
    print(f"\n⏰ Auto-running FridayTrader at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    subprocess.run(["python3", "/Users/vedantnogaja/Documents/FridayTrader/friday_trader_v3.py"],
                   env={**os.environ})

print("🤖 FridayTrader Auto-Scheduler Started!")
print("📅 Will run Monday-Friday at 10:00 AM Singapore time")
print("Press Ctrl+C to stop\n")

# Run once immediately on start
run_agent()

# Schedule Mon-Fri at 10am Singapore time
schedule.every().monday.at("22:00").do(run_agent)
schedule.every().tuesday.at("22:00").do(run_agent)
schedule.every().wednesday.at("22:00").do(run_agent)
schedule.every().thursday.at("22:00").do(run_agent)
schedule.every().friday.at("22:00").do(run_agent)

while True:
    schedule.run_pending()
    time.sleep(60)
