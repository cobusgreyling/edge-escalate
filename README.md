![edge-escalate](assets/header.png)

# edge-escalate

Stay on device, or pay for a cloud call. The choice is a score, not a model.

Each turn gets points for an unresolved task, a case the device has not seen, a high-risk fact, and a long utterance. The threshold is the value with the best escalate F1 on the train split. A tie goes to the higher threshold, which sends fewer turns to the cloud. The holdout is scored once, after that choice. No GPU and no API key.

The weights are in `edge_escalate/score.py`: unresolved 2, novel 2, high risk 3, and 1 more when the turn has at least 80 tokens. A cloud call is $0.002. That price is illustrative.

## Run

From the repo root. Python 3.11 or newer.

```bash
python3 -m edge_escalate fixtures/turns.jsonl
python3 -m edge_escalate --verbose fixtures/turns.jsonl
python3 -m unittest discover -s tests -v
```

## Score

```
split    n  precision  recall  escalations  cloud usd  saved vs always
-------  -  ---------  ------  -----------  ---------  ---------------
holdout  8       0.75    0.75            4     $0.008           $0.008

Threshold 4 was chosen on train by escalate F1. A cloud call is $0.002 illustrative.
```

Eight holdout turns. Four go to the cloud. Precision and recall for escalate are both 0.75. Against calling the cloud every time, the holdout saves $0.008.

The false escalate is `h-closed-novel-risk`: the case is new and high risk, but it is already closed, and the score still sends it. The false stay is `h-long-unresolved`: the caller retells a known open claim at length, and length alone does not clear the threshold.

## Turn shape

One JSON object per line. `split` is `train` or `holdout`. `label` is `stay` or `escalate`.

```json
{"id":"h-unresolved-risk","split":"holdout","label":"escalate","unresolved":true,"novel":false,"high_risk":true,"tokens":16,"text":"The open claim still has no incident date, and the amount is above the desk limit."}
```

## License

MIT
