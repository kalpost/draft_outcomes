import os, requests
from dotenv import load_dotenv
load_dotenv()
H = {"Authorization": f"Bearer {os.environ['CFBD_API_KEY']}"}
for yr in (2010, 2011):
    r = requests.get("https://api.collegefootballdata.com/games/players", headers=H, params={"year": yr, "week": 5})
    print(yr, r.status_code, len(r.json()) if r.ok else r.text[:100])