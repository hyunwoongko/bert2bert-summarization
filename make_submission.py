import pandas as pd

fname = "submission_25000"
file = open(fname + ".csv", "r", encoding="utf-8").read().splitlines()

ids, summaries = [], []
for line in file:
    line = line.split(",")
    news_id = line[0]
    summary = line[1:]
    summary = ",".join(summary)
    ids.append(news_id)
    summaries.append(summary)

data = {
    "id": ids,
    "summary": summaries
}

out = pd.DataFrame(data, columns=["id", "summary"], index=None)
out.set_index('id', inplace=True)
out.to_csv(fname + "_processed.csv")
