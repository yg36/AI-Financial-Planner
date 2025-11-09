#!/usr/bin/env python3
"""
MAFP CLI prototype v2
Improvements:
 - Step 1: Smarter intent detection using TF-IDF + cosine similarity
 - Step 2: Financial Health Score displayed at login and via /summary

Dependencies: pandas, numpy, matplotlib, scikit-learn
Run: python mafp_cli_v2.py
"""
import sys, os, json, getpass, math, datetime, random
from collections import defaultdict
from pprint import pprint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ML/NLP imports for Step 1
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------
# Simple "database"
# -------------------------
DATA_DIR = "mafp_data"
os.makedirs(DATA_DIR, exist_ok=True)
USERS_FILE = os.path.join(DATA_DIR, "users.json")
DECISION_LOG = os.path.join(DATA_DIR, "decision_logs.json")

# seed users (if not exist)
if not os.path.exists(USERS_FILE):
    users = {
        "alice": {"password": "password", "name": "Alice", "balances": {"checking": 5200, "savings": 8000}}
    }
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)
else:
    users = json.load(open(USERS_FILE))

if not os.path.exists(DECISION_LOG):
    with open(DECISION_LOG, "w") as f:
        json.dump([], f)

# -------------------------
# Mock transactions generator
# -------------------------
CATEGORIES = ["dining", "groceries", "transport", "entertainment", "rent", "utilities", "shopping", "travel"]
def generate_transactions(user, days=180, seed=42):
    random.seed(seed)
    rows = []
    today = datetime.date.today()
    for i in range(days):
        d = today - datetime.timedelta(days=i)
        # simulate 1-4 transactions per day
        for t in range(random.randint(1,4)):
            cat = random.choices(CATEGORIES, weights=[12,18,10,8,20,5,10,5])[0]
            amount = round(max(1.5, random.gauss(20 if cat not in ("rent","travel") else 150, 40)), 2)
            if cat == "rent" and d.day == 1:
                amount = 1200.0
            currency = "USD"
            rows.append({"date": d.isoformat(), "category": cat, "amount": amount, "currency": currency, "description": f"{cat} expense"})
    df = pd.DataFrame(rows)
    df.sort_values("date", inplace=True)
    return df

# save sample transactions for alice
TX_FILE = os.path.join(DATA_DIR, "alice_tx.csv")
if not os.path.exists(TX_FILE):
    df = generate_transactions("alice", days=365)
    df.to_csv(TX_FILE, index=False)

# -------------------------
# Decision logging helper
# -------------------------
def log_decision(user, session_id, agent, prompt, decision):
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "user": user,
        "session": session_id,
        "agent": agent,
        "prompt": prompt,
        "decision": decision
    }
    logs = json.load(open(DECISION_LOG))
    logs.append(entry)
    with open(DECISION_LOG, "w") as f:
        json.dump(logs, f, indent=2)
    # readable console summary
    print(f"[{agent.upper()}] {decision.get('summary','-')}")

# -------------------------
# Step 1: Smarter intent detection (TF-IDF + cosine similarity)
# -------------------------
intent_examples = {
    "spending_breakdown": [
        "show my spending",
        "how much did I spend last month",
        "spending analysis",
        "spending breakdown for the last 30 days",
        "show spending by category"
    ],
    "big_purchase": [
        "I want to buy a car",
        "save for a laptop",
        "help me plan a big purchase",
        "I need to save $30000 next year",
        "how to save for a house"
    ],
    "trip_plan": [
        "plan a trip to NYC",
        "vacation budget",
        "book hotels and flights",
        "plan a 4-day trip",
        "travel itinerary and budget"
    ]
}

# Build TF-IDF vectorizer on intent example phrases
_all_examples = []
for exs in intent_examples.values():
    _all_examples.extend(exs)
vectorizer = TfidfVectorizer().fit(_all_examples)

# Create mean vector for each intent (centroid)
intent_centroids = {}
for k, exs in intent_examples.items():
    mat = vectorizer.transform(exs)  # shape (n_examples, n_features)
    centroid = np.asarray(mat.mean(axis=0)).ravel()  # 1D numpy array
    intent_centroids[k] = centroid

def detect_intent(text):
    """Return intent string using cosine similarity against intent centroids.
       If no centroid has similarity above threshold, return 'unknown'."""
    vec = vectorizer.transform([text]).toarray().ravel()
    scores = {}
    for k, centroid in intent_centroids.items():
        sim = 0.0
        # handle zero vectors (rare)
        if np.linalg.norm(vec) > 0 and np.linalg.norm(centroid) > 0:
            sim = float(cosine_similarity(vec.reshape(1,-1), centroid.reshape(1,-1))[0,0])
        scores[k] = sim
    best = max(scores, key=scores.get)
    # threshold: require a minimum similarity to accept mapping
    if scores[best] < 0.18:
        return "unknown"
    return best

# -------------------------
# Router
# -------------------------
def router(user, session_id, query):
    intent = detect_intent(query)
    if intent == "spending_breakdown":
        return daily_spending_advisor(user, session_id, query)
    elif intent == "big_purchase":
        return big_purchase_planner(user, session_id, query)
    elif intent == "trip_plan":
        return trip_planning_assistant(user, session_id, query)
    else:
        # fallback: give suggestions + ask clarifying
        reply = {"text": "I wasn't sure which assistant you need. Try: 'Show spending last 30 days', 'I want to buy a $30K car next year', or 'Plan a 4-day NYC trip with $2000'."}
        log_decision(user, session_id, "router", query, {"summary": "unclear_intent", "scores": {k: round(float(v),3) for k,v in intent_centroids.items()}})
        return reply

# -------------------------
# Step 2: Financial Health Score
# -------------------------
def financial_health_score(user):
    """Compute a simple financial health score (0-100).
       Factors: savings vs avg monthly spend, spending volatility.
    """
    try:
        tx = pd.read_csv(TX_FILE, parse_dates=["date"])
    except Exception:
        return {"score": None, "reason": "no transactions available"}

    # monthly spend average (last 6 months)
    tx['month'] = tx['date'].dt.to_period('M')
    monthly_spend = tx.groupby('month')['amount'].sum().astype(float)
    if len(monthly_spend) == 0:
        return {"score": None, "reason": "insufficient data"}

    avg_spend = float(monthly_spend.tail(6).mean())
    volatility = float(tx['amount'].std())

    user_info = users.get(user, {})
    savings_balance = float(user_info.get("balances", {}).get("savings", 0))
    # compute ratio (savings relative to monthly spending)
    ratio = savings_balance / (avg_spend + 1e-9)  # avoid division by zero

    # Score heuristics:
    # Base 60, add points for larger ratio, subtract for volatility
    score = 60.0
    score += min(30.0, max(-20.0, (ratio - 3.0) * 3.0))   # reward higher savings ratio
    score -= min(30.0, max(0.0, volatility / 50.0))        # penalize high volatility
    score = max(0.0, min(100.0, score))
    label = "Excellent" if score >= 80 else ("Good" if score >= 65 else ("Fair" if score >= 45 else "Poor"))
    return {"score": round(score, 2), "label": label, "avg_monthly_spend": round(avg_spend,2), "savings_balance": round(savings_balance,2), "volatility": round(volatility,2)}

# -------------------------
# Agent: Daily Spending Advisor
# -------------------------
def daily_spending_advisor(user, session_id, query, days=30):
    tx = pd.read_csv(TX_FILE, parse_dates=["date"])
    cutoff = datetime.date.today() - datetime.timedelta(days=days)
    recent = tx[tx["date"].dt.date >= cutoff]
    # categorize (already categorized)
    cat_sum = recent.groupby("category")["amount"].sum().sort_values(ascending=False)
    total = cat_sum.sum()
    breakdown = [{"category": c, "amount": float(a), "pct": round(100*float(a)/total,2)} for c,a in cat_sum.items()] if total>0 else []

    # anomaly detection: simple z-score per category daily totals
    daily = recent.groupby([recent['date'].dt.date, 'category'])['amount'].sum().reset_index()
    anomalies = []
    for cat in daily['category'].unique():
        arr = daily[daily['category']==cat]['amount'].values
        if len(arr) < 5: continue
        mean = arr.mean(); std = arr.std()
        for val in arr:
            if std>0 and (val - mean)/std > 2.5:
                anomalies.append({"category": cat, "value": float(val), "mean": float(mean), "z": float((val-mean)/std)})

    # generate pie chart (if there is data)
    fig_path = os.path.join(DATA_DIR, f"{user}_spending_pie.png")
    if total > 0:
        plt.style.use('ggplot')
        cat_sum.plot.pie(y='amount', autopct='%1.1f%%', legend=False)
        plt.ylabel('')
        plt.title(f"{user} - Spending last {days} days (total ${total:.2f})")
        plt.tight_layout()
        plt.savefig(fig_path)
        plt.clf()
    else:
        fig_path = None

    summary = f"Spending total ${total:.2f} over last {days} days. Top categories: {', '.join([b['category'] for b in breakdown[:3]])}." if total>0 else "No spending in the selected period."
    decision = {"summary": summary, "breakdown": breakdown, "anomalies": anomalies, "chart": fig_path}
    log_decision(user, session_id, "daily_spending_advisor", query, decision)
    return {"text": summary, "breakdown": breakdown, "anomalies": anomalies, "chart": fig_path}

# -------------------------
# Agent: Big Purchase Planner
# -------------------------
def big_purchase_planner(user, session_id, query):
    # parse target amount and timeframe (naive)
    import re
    m = re.search(r"\$?([0-9,]+(?:\.[0-9]{1,2})?)", query.replace(',', ''))
    target_amount = float(m.group(1)) if m else None
    # look for timeframe (year => 12 months etc.)
    months = 12
    if "year" in query:
        months = 12
    if "next year" in query:
        months = 12
    m2 = re.search(r"in (\d+) months", query)
    if m2:
        months = int(m2.group(1))

    # load balances and transaction history to compute monthly savings
    tx = pd.read_csv(TX_FILE, parse_dates=["date"])
    tx['month'] = tx['date'].dt.to_period('M')
    monthly_spend = tx.groupby('month')['amount'].sum().astype(float)
    avg_spend = float(monthly_spend.tail(6).mean()) if len(monthly_spend) > 0 else 0.0

    # read user balances (simulate)
    user_info = users.get(user, {})
    savings_balance = float(user_info.get("balances", {}).get("savings", 0))

    # simple forecast: suggest monthly saving using heuristic plus small ML tweak:
    # use avg_spend to scale suggestion
    suggested_monthly_saving = max(50, round(0.12 * avg_spend + 75, 2))  # updated heuristic

    if target_amount is None:
        return {"text": "Please state a target amount, e.g., 'I want to buy a $30000 car next year'."}

    needed = max(0.0, target_amount - savings_balance)
    months_needed = math.ceil(needed / suggested_monthly_saving) if suggested_monthly_saving>0 else None

    # generate timeline chart
    months_plot = list(range(0, min(60, max(months_needed or 0, months))))
    timeline = [savings_balance + suggested_monthly_saving * m for m in months_plot]
    fig_path = os.path.join(DATA_DIR, f"{user}_savings_timeline.png")
    plt.style.use('ggplot')
    plt.plot(months_plot, timeline, marker='o')
    plt.xlabel("Months")
    plt.ylabel("Projected Savings (USD)")
    plt.title(f"Savings projection to reach ${target_amount:,.0f}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.clf()

    # recommendation: adjust savings or target timeframe
    recommendation = {}
    if months_needed is not None and months_needed <= months:
        recommendation['can_meet'] = True
        recommendation['plan'] = f"Save ${suggested_monthly_saving:.2f}/month and you'll reach ${target_amount:.2f} in {months_needed} months."
    else:
        recommendation['can_meet'] = False
        recommendation['plan'] = f"At ${suggested_monthly_saving:.2f}/month it will take {months_needed} months; consider increasing monthly savings or lowering target."

    decision = {
        "summary": f"Target: ${target_amount:.2f}, savings_balance: ${savings_balance:.2f}, monthly_save: ${suggested_monthly_saving:.2f}, months_needed: {months_needed}",
        "details": {"target": target_amount, "savings_balance": savings_balance, "monthly_saving": suggested_monthly_saving, "months_needed": months_needed},
        "chart": fig_path
    }
    log_decision(user, session_id, "big_purchase_planner", query, decision)
    return {"text": recommendation['plan'], "chart": fig_path, "details": decision["details"]}

# -------------------------
# Agent: Trip Planning Assistant (simulated)
# -------------------------
def trip_planning_assistant(user, session_id, query):
    import re
    m_days = re.search(r"(\d+)[- ]?day", query)
    days = int(m_days.group(1)) if m_days else 3
    m_budget = re.search(r"\$?([0-9,]+)", query.replace(',', ''))
    budget = float(m_budget.group(1)) if m_budget else 1500.0
    city = "NYC" if "nyc" in query.lower() else "destination"

    hotels = [
        {"name": "Budget Inn", "price_per_night": 80},
        {"name": "Comfort Suites", "price_per_night": 150},
        {"name": "Boutique Hotel", "price_per_night": 280},
    ]
    activities = [
        {"name": "City Tour", "price": 40},
        {"name": "Broadway Show", "price": 120},
        {"name": "Museum Pass", "price": 30}
    ]
    estimated_hotel_total = [(h, h['price_per_night']*days) for h in hotels]
    best_plan = None
    for hotel, hotel_cost in estimated_hotel_total:
        activities_cost = activities[0]['price'] + activities[1]['price']
        airfare = 300
        total = hotel_cost + activities_cost + airfare + 100
        if total <= budget:
            best_plan = {"hotel": hotel, "activities": activities[:2], "airfare": airfare, "total": total}
            break
    if best_plan is None:
        h, hotel_cost = min(estimated_hotel_total, key=lambda x: x[1])
        total = hotel_cost + activities[0]['price'] + 300 + 100
        best_plan = {"hotel": h, "activities": [activities[0]], "airfare": 300, "total": total, "over_budget": total - budget}

    labels = ["hotel", "activities", "airfare", "misc"]
    values = [best_plan['hotel']['price_per_night']*days, sum(a['price'] for a in best_plan['activities']), best_plan['airfare'], 100]
    fig_path = os.path.join(DATA_DIR, f"{user}_trip_{city}.png")
    plt.style.use('ggplot')
    plt.pie(values, labels=labels, autopct='%1.1f%%')
    plt.title(f"{city} trip budget (total ${best_plan['total']:.2f})")
    plt.savefig(fig_path)
    plt.clf()

    decision = {"summary": f"Planned {days}-day {city} trip cost ${best_plan['total']:.2f}", "plan": best_plan, "chart": fig_path}
    log_decision(user, session_id, "trip_planner", query, decision)
    text = f"Planned {days}-day {city} trip. Estimated total ${best_plan['total']:.2f}."
    if best_plan.get("over_budget"):
        text += f" Note: Over budget by ${best_plan['over_budget']:.2f}."
    return {"text": text, "plan": best_plan, "chart": fig_path}

# -------------------------
# CLI Interface
# -------------------------
def authenticate():
    print("=== MAFP CLI v2 ===")
    uname = input("Username: ").strip()
    pwd = getpass.getpass("Password: ")
    if uname in users and users[uname]["password"] == pwd:
        info = users[uname]
        print(f"Welcome back, {info.get('name', uname)}")
        # show financial health on login
        fh = financial_health_score(uname)
        if fh.get("score") is not None:
            print(f"💡 Financial Health Score: {fh['score']}/100 ({fh['label']}) — Savings ${fh['savings_balance']}, Avg monthly spend ${fh['avg_monthly_spend']}")
        else:
            print("💡 Financial Health Score: unavailable —", fh.get("reason", "insufficient data"))
        return uname
    else:
        print("Invalid credentials. (Try username 'alice' password 'password')")
        return None

def show_help():
    print("""
Examples:
  - Show me my spending breakdown for the last 30 days
  - I want to buy a $30000 car next year
  - Plan a 4-day NYC trip with a $2000 budget
Commands:
  /exit     - quit
  /help     - show this
  /logs     - show decision log summary
  /summary  - quick finance summary & health score
""")

def show_summary(user):
    fh = financial_health_score(user)
    print("\n--- Quick Summary ---")
    if fh.get("score") is not None:
        print(f"Financial Health: {fh['score']}/100 ({fh['label']})")
        print(f" - Savings balance: ${fh['savings_balance']}")
        print(f" - Avg monthly spend (last 6 months): ${fh['avg_monthly_spend']}")
        print(f" - Spending volatility (std): ${fh['volatility']}")
    else:
        print("Financial health: unavailable —", fh.get("reason",""))

    # top categories last 30 days
    tx = pd.read_csv(TX_FILE, parse_dates=["date"])
    cutoff = datetime.date.today() - datetime.timedelta(days=30)
    recent = tx[tx["date"].dt.date >= cutoff]
    if recent.empty:
        print("No transactions in the last 30 days.")
        return
    cat_sum = recent.groupby("category")["amount"].sum().sort_values(ascending=False)
    print("\nTop categories (30 days):")
    for c,a in cat_sum.items():
        pct = 100*float(a)/cat_sum.sum()
        bar = "#" * int(pct // 2)
        print(f"  {c:12} ${a:8.2f}  {pct:5.1f}%  {bar}")

def main():
    user = None
    while not user:
        user = authenticate()
        if not user:
            if input("Try again? (y/n): ").lower() != 'y':
                print("Exiting.")
                return
    session_id = f"{user}_{int(datetime.datetime.now().timestamp())}"
    show_help()
    while True:
        q = input("\nYou: ").strip()
        if not q:
            continue
        if q == "/exit":
            print("Goodbye.")
            break
        if q == "/help":
            show_help()
            continue
        if q == "/logs":
            logs = json.load(open(DECISION_LOG))
            print(f"Total decisions logged: {len(logs)} (showing last 3)")
            for e in logs[-3:]:
                print(f"- {e['timestamp']} | {e['agent']} | {e['decision'].get('summary','-')}")
            continue
        if q == "/summary":
            show_summary(user)
            continue

        # route the query
        try:
            out = router(user, session_id, q)
            # print output elegantly
            if isinstance(out, dict):
                print("\n--- Assistant reply ---")
                print(out.get("text","(no text)"))
                if out.get("breakdown"):
                    print("\nCategory breakdown:")
                    for b in out["breakdown"][:10]:
                        bar = "#" * int(b['pct'] // 2)
                        print(f"  {b['category']:12} ${b['amount']:8.2f}  {b['pct']:5.1f}%  {bar}")
                if out.get("chart"):
                    print(f"\nChart saved to: {out['chart']}")
                if out.get("details"):
                    print("\nDetails:")
                    pprint(out["details"])
                if out.get("plan"):
                    print("\nPlan details:")
                    pprint(out["plan"])
            else:
                print(out)
        except Exception as e:
            # log and show friendly error
            print("Sorry — something went wrong. The error has been logged.")
            log_decision(user, session_id, "router_error", q, {"summary": str(e)})
            # continue loop

if __name__ == "__main__":
    main()
