"""
Project 9: Personal Expense Tracker
Libraries: NumPy, Pandas, Matplotlib
Year: 2024
"""
import calendar
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, DateFormatter

# 1. Reproducible data generation
rng = np.random.default_rng(9)
year = 2024
months = pd.date_range("2024-01-01", "2024-12-01", freq="MS")
categories = ["Rent", "Food", "Transport", "Entertainment", "Bills", "Shopping"]
budgets = pd.Series({
    "Rent": 850, "Food": 500, "Transport": 190,
    "Entertainment": 120, "Bills": 220, "Shopping": 480
}, dtype=float)
records = []

# Rent on the first of every month
for month in months:
    records.append({"Date": month, "Category": "Rent",
                    "Amount": 850.00, "Description": "Monthly rent"})

# Three bills per month
bill_items = [
    ("Electricity bill", 65, 125),
    ("Internet bill", 35, 55),
    ("Water bill", 20, 40),
]
for month in months:
    for description, low, high in bill_items:
        records.append({
            "Date": month + pd.Timedelta(days=int(rng.integers(2, 25))),
            "Category": "Bills",
            "Amount": round(float(rng.uniform(low, high)), 2),
            "Description": description
        })

# Regular monthly purchase counts
monthly_counts = {"Food": 26, "Transport": 14,
                  "Entertainment": 6, "Shopping": 4}
for month in months:
    for category, count in monthly_counts.items():
        if category == "Shopping" and month.month == 11:
            count = 7
        for _ in range(count):
            day = int(rng.integers(1, (month + pd.offsets.MonthEnd(0)).day + 1))
            if category in ("Food", "Transport"):
                amount = rng.lognormal(mean=2.65, sigma=0.55)
            elif category == "Entertainment":
                amount = rng.lognormal(mean=3.1, sigma=0.65)
            else:
                amount = rng.lognormal(mean=3.5, sigma=0.6)
            records.append({
                "Date": month + pd.Timedelta(days=day - 1),
                "Category": category,
                "Amount": round(float(amount), 2),
                "Description": f"{category} purchase"
            })

# Required one-off expenses
one_offs = [
    ("2024-03-14", "Shopping", 1450, "Laptop"),
    ("2024-06-20", "Transport", 1150, "Car repair"),
    ("2024-08-03", "Transport", 1250, "Flights"),
    ("2024-10-09", "Bills", 1100, "Insurance"),
    ("2024-11-29", "Shopping", 650, "Black Friday purchase 1"),
    ("2024-11-29", "Shopping", 650, "Black Friday purchase 2"),
    ("2024-11-29", "Shopping", 650, "Black Friday purchase 3"),
]
for date, category, amount, description in one_offs:
    records.append({"Date": pd.Timestamp(date), "Category": category,
                    "Amount": float(amount), "Description": description})

# 2. Build and clean the transactions DataFrame
df = pd.DataFrame(records)
df["Date"] = pd.to_datetime(df["Date"])
df["Amount"] = df["Amount"].round(2)
df = df.sort_values(["Date", "Category"]).reset_index(drop=True)
df["Month"] = df["Date"].dt.to_period("M").astype(str)

# 3. Monthly totals by category (pivot table)
monthly_category = df.pivot_table(
    index="Month", columns="Category", values="Amount",
    aggfunc="sum", fill_value=0
).reindex(
    index=[m.strftime("%Y-%m") for m in months],
    columns=categories, fill_value=0
)
monthly_totals = monthly_category.sum(axis=1)

# 4. Annual category totals and category shares
annual_category = (
    df.groupby("Category")["Amount"].sum()
      .reindex(categories, fill_value=0)
      .sort_values(ascending=False)
)
annual_total = float(df["Amount"].sum())
transaction_count = len(df)
average_monthly = annual_total / 12
category_share = annual_category / annual_total * 100

# 5. Compare average monthly actual spending with budget
average_actual = monthly_category.mean()
budget_utilization = average_actual / budgets * 100
months_over_budget = (monthly_category > budgets).sum()

# 6. Five costliest days and cumulative spending
top_days = (
    df.groupby("Date")["Amount"].sum()
      .sort_values(ascending=False).head(5)
      .rename("Total Spend").reset_index()
)
daily = df.groupby("Date")["Amount"].sum().reindex(
    pd.date_range("2024-01-01", "2024-12-31", freq="D"),
    fill_value=0
)
cumulative = daily.cumsum()

# 7. Four-panel dashboard
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
ax1, ax2, ax3, ax4 = axes.flatten()
x = np.arange(len(monthly_category))
bottom = np.zeros(len(monthly_category))

for category in categories:
    values = monthly_category[category].to_numpy()
    ax1.bar(x, values, bottom=bottom, label=category)
    bottom += values
ax1.set_title("Monthly Spending by Category")
ax1.set_xlabel("Month")
ax1.set_ylabel("Spending ($)")
ax1.set_xticks(x)
ax1.set_xticklabels([calendar.month_abbr[m.month] for m in months])
ax1.legend(fontsize=8, ncol=3)
ax1.grid(axis="y", alpha=0.25)

share_sorted = category_share.sort_values()
bars = ax2.barh(share_sorted.index, share_sorted.values)
ax2.set_title("Share of Year's Spending")
ax2.set_xlabel("Share of total (%)")
for bar, category in zip(bars, share_sorted.index):
    ax2.text(bar.get_width() + 0.5,
             bar.get_y() + bar.get_height() / 2,
             f"{bar.get_width():.1f}% (${annual_category[category]:,.0f})",
             va="center", fontsize=8)
ax2.grid(axis="x", alpha=0.25)

ax3.plot(cumulative.index, cumulative.values,
         label="Actual cumulative spend")
budget_pace = np.linspace(0, budgets.sum() * 12, len(cumulative))
ax3.plot(cumulative.index, budget_pace, linestyle="--",
         label="Budgeted pace")
ax3.set_title("Cumulative Spending Through the Year")
ax3.set_xlabel("Date")
ax3.set_ylabel("Total spend ($)")
ax3.xaxis.set_major_locator(MonthLocator())
ax3.xaxis.set_major_formatter(DateFormatter("%b"))
for _, row in top_days.iterrows():
    date = row["Date"]
    y = cumulative.loc[date]
    ax3.scatter(date, y, s=22)
    ax3.annotate(f"{date:%d %b}\\n${row['Total Spend']:,.0f}",
                 (date, y), textcoords="offset points",
                 xytext=(0, 7), ha="center", fontsize=7)
ax3.legend(fontsize=8)
ax3.grid(alpha=0.25)

cat_order = budgets.index.tolist()
pos = np.arange(len(cat_order))
width = 0.38
ax4.barh(pos - width / 2, budgets.values, height=width,
         label="Monthly budget")
ax4.barh(pos + width / 2, average_actual.reindex(cat_order).values,
         height=width, label="Average actual")
ax4.set_yticks(pos)
ax4.set_yticklabels(cat_order)
ax4.set_title("Monthly Budget vs Average Actual")
ax4.set_xlabel("Dollars per month ($)")
ax4.legend(fontsize=8)
ax4.grid(axis="x", alpha=0.25)

fig.suptitle("Personal Expense Tracker — 2024",
             fontsize=16, fontweight="bold")
fig.tight_layout(rect=[0, 0.02, 1, 0.96])
fig.savefig("chart.png", dpi=120, bbox_inches="tight")
plt.show()

# 8. Print summary report
print("\\nPERSONAL EXPENSE TRACKER — 2024")
print("=" * 45)
print(f"Transactions: {transaction_count}")
print(f"Annual spending: ${annual_total:,.2f}")
print(f"Average monthly spending: ${average_monthly:,.2f}")
print(f"Priciest month: {monthly_totals.idxmax()} "
      f"(${monthly_totals.max():,.2f})")
print(f"Cheapest month: {monthly_totals.idxmin()} "
      f"(${monthly_totals.min():,.2f})")

print("\\nMONTHLY SPENDING")
print(monthly_totals.round(2).to_string())

print("\\nCATEGORY TOTALS AND SHARE")
category_report = pd.DataFrame({
    "Annual Total": annual_category,
    "Share (%)": category_share
})
print(category_report.round(2).to_string())

print("\\nBUDGET CHECK")
budget_report = pd.DataFrame({
    "Budget per Month": budgets,
    "Average Actual": average_actual,
    "Budget Used (%)": budget_utilization,
    "Months Over Budget": months_over_budget
})
print(budget_report.round(2).to_string())

print("\\nFIVE COSTLIEST DAYS")
print(top_days.to_string(index=False, formatters={
    "Total Spend": lambda value: f"${value:,.2f}"
}))

# 9. Save machine-readable results
results = {
    "year": 2024,
    "transaction_count": int(transaction_count),
    "annual_spending": round(annual_total, 2),
    "average_monthly_spending": round(average_monthly, 2),
    "priciest_month": {
        "month": str(monthly_totals.idxmax()),
        "amount": round(float(monthly_totals.max()), 2)
    },
    "cheapest_month": {
        "month": str(monthly_totals.idxmin()),
        "amount": round(float(monthly_totals.min()), 2)
    },
    "category_totals": {
        str(k): round(float(v), 2) for k, v in annual_category.items()
    },
    "category_share_percent": {
        str(k): round(float(v), 2) for k, v in category_share.items()
    },
    "average_monthly_spending_by_category": {
        str(k): round(float(v), 2) for k, v in average_actual.items()
    },
    "budget_utilization_percent": {
        str(k): round(float(v), 2) for k, v in budget_utilization.items()
    },
    "months_over_budget": {
        str(k): int(v) for k, v in months_over_budget.items()
    },
    "top_five_costliest_days": [
        {"date": row["Date"].strftime("%Y-%m-%d"),
         "amount": round(float(row["Total Spend"]), 2)}
        for _, row in top_days.iterrows()
    ],
    "top_five_days_share_percent": round(
        float(top_days["Total Spend"].sum() / annual_total * 100), 2
    ),
    "total_spending_by_month": {
        str(k): round(float(v), 2) for k, v in monthly_totals.items()
    }
}
with open("results.json", "w", encoding="utf-8") as file:
    json.dump(results, file, indent=2)

df.drop(columns=["Month"]).to_csv("transactions_2024.csv", index=False)
print("\\nSaved chart.png, results.json and transactions_2024.csv")
