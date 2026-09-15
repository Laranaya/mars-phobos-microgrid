# phobosgrid

Demand-aware solar microgrid control for a Mars settlement.

Built for the GirlsWhoML x PhysicsX Mars Hackathon. This project is in the **Life Support & Resource Systems** track.

## What it does

phobosgrid is a Streamlit dashboard for a Mars settlement microgrid. It combines real orbital geometry with lightweight machine learning to answer one operational question:

**When Phobos dims the Sun, how should the settlement respond?**

The app shows:

- real SPICE-based Phobos transit timing and eclipse depth
- a pvlib-inspired solar baseline for the current Martian time
- a synthetic-demand RandomForestRegressor for settlement power demand
- a RandomForestClassifier that turns predicted transit depth + demand into an action level
- an optional 3D Plotly Mars / Phobos scene driven by real SPICE positions

## How the website works

The dashboard is intentionally simple:

1. The sidebar sets the moment in Martian time, plus whether the greenhouse is active.
2. `transit.py` uses NASA SPICE kernels to check whether Phobos is transiting the Sun at that moment and how much of the solar disk is blocked.
3. `solar.py` estimates the normal solar output for the same time.
4. `demand_model.py` predicts how much power the settlement needs.
5. `risk_model.py` combines predicted blockage + predicted demand and outputs the operating mode.
6. The main panel turns those results into a visual decision aid: metrics, risk label, transit warning, charts, and an optional 3D scene.

In other words, the sliders do not just change the picture. They change the actual control decision.

## Why this is a good Mars-city prototype

Mars settlements need closed-loop resource systems before they need novelty. Power is the backbone of life support, greenhouse operation, communications, heating, and battery buffering. Phobos transits are a real, predictable cause of short solar dips, so this project demonstrates a practical control loop for a first settlement:

1. physics tells us when solar input changes
2. a baseline estimates normal output
3. ML predicts demand
4. ML recommends the operating response

That makes the ML decision useful instead of decorative.

## ML usage

This project uses two small scikit-learn models:

- **Demand model**: `RandomForestRegressor`
  - Inputs: hour of day, greenhouse active flag, synthetic life-support baseline
  - Output: predicted settlement demand in kW

- **Risk model**: `RandomForestClassifier`
  - Inputs: predicted percent of solar disk blocked, predicted demand
  - Output:
    - `0` = Normal
    - `1` = Soft Curtailment
    - `2` = Critical Buffer

The models are trained on synthetic data because there is no real Mars settlement dataset yet.

## Real data and physics

- **SpiceyPy / NASA SPICE** is used for real planetary geometry and Phobos transits.
- The transit logic is in `transit.py` and uses the downloaded kernels in `kernels/`.
- The solar baseline is in `solar.py`.
- The 3D scene in `app.py` uses real SPICE positions via `spice.spkpos("PHOBOS", ...)` and `spice.spkpos("DEIMOS", ...)`.

## Demo flow

For the 3-minute presentation:

1. Show the headline dashboard.
2. Point out the solar output, predicted demand, and risk level.
3. Move the hour slider to show the SPICE-driven transit timing and the real dip.
4. Open the 3D Mars / Phobos panel as a visual bonus.
5. Close with the decision logic: predictive demand plus transit depth drives operational response.

## What the controls do

- **Hour of day**: picks the Martian hour the models and SPICE geometry should evaluate.
- **Jump to Phobos transit peak**: jumps the dashboard to the strongest transit moment for the demo.
- **Greenhouse active**: switches the demand model into a higher-load operating mode.
- **3D Mars / Phobos view**: a visual bonus that shows real SPICE positions for the current time.

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the app

```bash
streamlit run app.py
```

If `streamlit` is not on your PATH, use:

```bash
C:/Python313/python.exe -m streamlit run app.py
```

### 3. Open the dashboard

Streamlit will print a local URL, usually:

```text
http://localhost:8501
```

If that port is busy, use another one:

```bash
C:/Python313/python.exe -m streamlit run app.py --server.port 8502
```

## Included files

- `app.py` - Streamlit dashboard
- `transit.py` - SPICE transit geometry and eclipse depth
- `solar.py` - solar baseline estimate
- `demand_model.py` - demand regressor
- `risk_model.py` - risk classifier
- `kernels/` - SPICE kernels required by the transit code
- `test_spice.py`, `test_phobos.py` - simple SPICE validation scripts

## Notes for judges

- This is a solo build.
- The demand data is synthetic by design.
- The core value is the control decision, not the model complexity.
- The 3D scene is optional presentation polish; the main technical contribution is the SPICE + ML control loop.

## One-line summary

A demand-aware Mars solar microgrid that uses real Phobos transit physics and two small ML models to recommend when to curtail load or buffer on batteries.
