# Airline Delay Propagation — Regression Model

Predicts how flight delays propagate through an aircraft's daily schedule.

## The Problem

When an aircraft arrives late, the next flight on that same plane is at risk
of departing late too. This model predicts how much delay will carry over,
based on the upstream delay, turnaround conditions, and schedule context.

## Data

- **Source:** Bureau of Transportation Statistics (BTS) On-Time Performance data
- **Airline:** Delta Air Lines (DL)
- **Period:** 23 months (Jan 2024 – Dec 2025, excluding Dec 2024)
- **Unit of analysis:** Flight pairs — two consecutive flights on the same aircraft

## Features (9)

| Feature | Description |
|---|---|
| `turnaround_airport` | Airport where the aircraft turns around (categorical) |
| `dest_j` | Destination of the downstream flight (categorical) |
| `month` | Month of the year |
| `day_of_week` | Day of the week |
| `next_dep_hour` | Scheduled departure hour of the downstream flight |
| `upstream_arr_delay_min` | Arrival delay of the incoming flight (minutes) |
| `scheduled_turnaround_min` | Planned buffer between flights (minutes) |
| `actual_available_turnaround_min` | Actual time remaining before next departure (minutes) |
| `crs_elapsed_time_j` | Scheduled flight duration of the downstream leg (minutes) |

## Targets (2)

| Target | Description |
|---|---|
| `target_dep_delay_min` | Departure delay of the downstream flight (minutes) |
| `target_arr_delay_min` | Arrival delay of the downstream flight (minutes) |

## Model

- **Algorithm:** CatBoost Regressor (MultiRMSE)
- **Split:** Time-based 80/20 (no data leakage)

## Project Structure

```
airline-reg/
├── datasets/                  # Raw BTS monthly CSVs
├── data_prep/
│   ├── generate_flight_pairs.py
│   └── flight_pairs.csv       # Generated training dataset
├── models/
│   └── train_model_cb.py      # CatBoost training script
├── results/
│   ├── catboost_model.cbm     # Trained model
│   ├── metrics.csv
│   ├── feature_importance.csv
│   └── predictions.csv
└── README.md
```

## How to Run

```bash
# Step 1: Build the flight pair dataset
python data_prep/generate_flight_pairs.py

# Step 2: Train the model
python models/train_model_cb.py
```
