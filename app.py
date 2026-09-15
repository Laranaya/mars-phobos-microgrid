"""Mars microgrid mission-control dashboard."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import spiceypy as spice
import streamlit.components.v1 as components
import streamlit as st

from demand_model import predict_demand
from risk_model import RISK_LABELS, predict_risk
from solar import expected_solar_output_pct
from transit import TRANSIT_PEAK_UTC, get_sun_geometry, get_transit_info, load_kernels

st.set_page_config(
    page_title="phobosgrid // Mars Microgrid",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="expanded",
)

RISK_COLOR = {0: "#3DDC97", 1: "#FFB020", 2: "#FF4B4B"}
AMBER = "#FF8C32"
BG = "#0B0B0C"
VISUAL_SCALE = 0.001
MARS_RADIUS_KM = 3396.19

st.markdown(
    f"""
    <style>
            header[data-testid="stHeader"], div[data-testid="stToolbar"], div[data-testid="stDecoration"] {{ display: none; }}
            .block-container {{ padding-top: 0.85rem; }}
      .stApp {{ background: radial-gradient(1200px 600px at 20% -10%, #2a1508 0%, {BG} 45%); }}
      html, body, [class*="st-"] {{ color: #F5E6D3; }}
      [data-testid="stMetricValue"] {{
        font-family: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
        font-variant-numeric: tabular-nums;
        color: {AMBER};
      }}
      [data-testid="stMetricLabel"] {{ letter-spacing: 0.12em; text-transform: uppercase; font-size: 0.72rem; }}
      .badge {{
                                display: inline-block; width: fit-content; max-width: 100%; box-sizing: border-box;
                                padding: 0.35rem 0.7rem; border: 1px solid {AMBER};
                                font-family: ui-monospace, monospace; letter-spacing: 0.1em; font-size: 0.72rem;
        color: {AMBER};
      }}
            .sol-badge {{
                display: flex; flex-direction: column; gap: 0.05rem; width: 100%;
                margin: 0.15rem 0 0.85rem; white-space: normal; line-height: 1.05;
            }}
            .sol-badge .label {{ font-size: 0.6rem; letter-spacing: 0.2em; opacity: 0.8; }}
            .sol-badge .value {{ font-size: 0.78rem; letter-spacing: 0.06em; word-break: break-word; }}
      .risk-pill {{
        font-family: ui-monospace, monospace; font-size: 1.35rem; font-weight: 700;
        letter-spacing: 0.08em; padding: 0.6rem 0.9rem; border-radius: 4px;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading NAIF kernels…")
def _boot():
    load_kernels()
    return True


_boot()

LIVE_NOW = datetime.now(timezone.utc)
MISSION_DAY = LIVE_NOW.date()

st.sidebar.markdown("**MISSION // EQUATORIAL HAB-01**")
st.sidebar.caption("Site: 0°N, 0°E  ·  Frame: IAU_MARS  ·  Kernels: de440s + mar099s")
st.sidebar.markdown(
    f"<div class='badge sol-badge'><div class='label'>SOL DATE</div><div class='value'>{MISSION_DAY.isoformat()}</div></div>",
    unsafe_allow_html=True,
)

live_mode = st.sidebar.toggle("Live mode", value=True)
st.sidebar.caption("Live mode keeps the clock current without refreshing the page.")

current_time = datetime.now(timezone.utc) if live_mode else LIVE_NOW

hour = st.sidebar.slider("Hour of day (UTC)", 0, 23, TRANSIT_PEAK_UTC.hour, disabled=live_mode)
jump = st.sidebar.toggle("Jump to Phobos transit peak", value=True)
greenhouse = st.sidebar.toggle("Greenhouse active", value=True)
st.sidebar.caption("Transit peak: 12:24:26 UTC. Keep jump-to-peak on for the live demo.")

if live_mode:
    when = current_time
elif jump:
    when = TRANSIT_PEAK_UTC
else:
    when = datetime(MISSION_DAY.year, MISSION_DAY.month, MISSION_DAY.day, hour, 0, 0, tzinfo=timezone.utc)

is_transiting, percent_blocked = get_transit_info(when)
solar_clear = expected_solar_output_pct(when)
solar_now = solar_clear * (1.0 - percent_blocked / 100.0)
demand_kw = predict_demand(when.hour + when.minute / 60.0, greenhouse)
risk_level, confidence, proba = predict_risk(percent_blocked, demand_kw)
sun = get_sun_geometry(when)

st.markdown("# phobosgrid")
st.caption(
    "Demand-aware solar microgrid controller  ·  Phobos transit physics (SPICE)  "
    "·  Demand regressor  ·  Risk classifier"
)
st.markdown(
    f"<span class='badge'>T+ {when.strftime('%Y-%m-%d %H:%M:%S')} UTC</span>  "
    f"<span class='badge'>SUN EL {sun['elevation_deg']:+5.1f}°</span>  "
    f"<span class='badge'>MARS–SUN {sun['r_au']:.3f} AU</span>",
    unsafe_allow_html=True,
)

components.html(
        """
        <div style="
            margin: 0.6rem 0 0.9rem;
            padding: 0.7rem 1rem;
            border: 1px solid rgba(255, 140, 50, 0.35);
            border-radius: 12px;
            background: linear-gradient(90deg, rgba(255,140,50,0.12), rgba(255,140,50,0.03));
            box-shadow: 0 8px 24px rgba(0,0,0,0.18);
            font-family: IBM Plex Mono, ui-monospace, monospace;
            color: #F5E6D3;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
        ">
            <div style="display:flex;flex-direction:column;gap:0.15rem;">
                <div style="font-size:0.62rem;letter-spacing:0.22em;opacity:0.72;text-transform:uppercase;">Live UTC clock</div>
                <div style="font-size:0.78rem;opacity:0.82;">Watching the time that drives the solar geometry and moon positions</div>
            </div>
            <div id="phobosgrid-top-clock" style="font-size:1.35rem;letter-spacing:0.08em;color:#FF8C32;font-weight:700;white-space:nowrap;">--:--:-- UTC</div>
        </div>
        <script>
            const pad = (value) => String(value).padStart(2, '0');
            function tickPhobosGridClock() {
                const now = new Date();
                const text = `${pad(now.getUTCHours())}:${pad(now.getUTCMinutes())}:${pad(now.getUTCSeconds())} UTC`;
                const el = document.getElementById('phobosgrid-top-clock');
                if (el) el.textContent = text;
            }
            tickPhobosGridClock();
            setInterval(tickPhobosGridClock, 1000);
        </script>
        """,
        height=92,
)

if live_mode:
        st.caption("Live view is synced to the current UTC clock. The scene animates in-browser, so Mars, Phobos, and the control decision keep moving without page refreshes.")
        components.html(
                """
                <div style="display:flex;align-items:center;gap:0.8rem;margin:0.35rem 0 0.1rem;font-family:IBM Plex Mono, ui-monospace, monospace;color:#F5E6D3;">
                    <div style="font-size:0.62rem;letter-spacing:0.2em;opacity:0.75;text-transform:uppercase;">Live UTC</div>
                    <div id="phobosgrid-clock" style="font-size:1.05rem;letter-spacing:0.08em;color:#FF8C32;">--:--:-- UTC</div>
                </div>
                <script>
                    const pad = (value) => String(value).padStart(2, '0');
                    function tickClock() {
                        const now = new Date();
                        const text = `${pad(now.getUTCHours())}:${pad(now.getUTCMinutes())}:${pad(now.getUTCSeconds())} UTC`;
                        const el = document.getElementById('phobosgrid-clock');
                        if (el) el.textContent = text;
                    }
                    tickClock();
                    setInterval(tickClock, 1000);
                </script>
                """
            )

r1c1, r1c2 = st.columns(2)
r1c1.metric("Solar output", f"{solar_now:5.1f} %", delta=f"{-percent_blocked:.1f} pt transit" if percent_blocked else "no dip")
r1c2.metric("Predicted demand", f"{demand_kw:5.1f} kW")
r2c1, r2c2 = st.columns(2)
r2c1.metric("Disk blocked", f"{percent_blocked:5.1f} %")
with r2c2:
    st.markdown("RISK LEVEL")
    st.markdown(
        f"<div class='risk-pill' style='background:{RISK_COLOR[risk_level]}22;color:{RISK_COLOR[risk_level]};border:1px solid {RISK_COLOR[risk_level]}'>"
        f"{risk_level} · {RISK_LABELS[risk_level].upper()}</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        f"confidence {confidence * 100:.0f}%  ·  "
        f"P0={proba[0]:.2f}  P1={proba[1]:.2f}  P2={proba[2]:.2f}"
    )

if is_transiting:
    st.warning(
        f"PHOBOS TRANSIT IN PROGRESS — {percent_blocked:.1f}% of the solar disk occulted. "
        + (
            "Activate battery buffer and shed non-critical lighting."
            if risk_level == 2
            else "Dim habitat lights; hold ISRU / greenhouse surge loads."
            if risk_level == 1
            else "Margin remaining; log the event."
        )
    )
else:
    st.info("No Phobos occultation at this timestamp. Solar baseline from pvlib + SPICE zenith.")


@st.cache_data(show_spinner=False)
def _visual_mission_time(hour_of_day: int) -> datetime:
    return datetime(MISSION_DAY.year, MISSION_DAY.month, MISSION_DAY.day, int(hour_of_day), 0, 0, tzinfo=timezone.utc)


@st.cache_data(show_spinner=False)
def _starfield(seed: int = 17, count: int = 180) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    azimuth = rng.uniform(0.0, 2.0 * np.pi, count)
    inclination = np.arccos(rng.uniform(-1.0, 1.0, count))
    radius = rng.uniform(45.0, 72.0, count)
    x = radius * np.sin(inclination) * np.cos(azimuth)
    y = radius * np.sin(inclination) * np.sin(azimuth)
    z = radius * np.cos(inclination)
    return pd.DataFrame({"x": x, "y": y, "z": z})


def _sphere_surface(center: tuple[float, float, float], radius: float, color: str, name: str) -> go.Surface:
    phi = np.linspace(0.0, 2.0 * np.pi, 36)
    theta = np.linspace(0.0, np.pi, 18)
    x = center[0] + radius * np.outer(np.cos(phi), np.sin(theta))
    y = center[1] + radius * np.outer(np.sin(phi), np.sin(theta))
    z = center[2] + radius * np.outer(np.ones_like(phi), np.cos(theta))
    return go.Surface(
        x=x,
        y=y,
        z=z,
        name=name,
        showscale=False,
        hoverinfo="skip",
        colorscale=[[0.0, color], [1.0, color]],
        surfacecolor=np.zeros_like(x),
        opacity=1.0,
        lighting=dict(ambient=0.88, diffuse=0.72, roughness=0.8, specular=0.12, fresnel=0.18),
    )


def _mars_moon_positions(when_utc: datetime) -> dict[str, np.ndarray]:
    load_kernels()
    et = spice.datetime2et(when_utc)
    phobos_vec, _ = spice.spkpos("PHOBOS", et, "J2000", "NONE", "MARS")
    deimos_vec, _ = spice.spkpos("DEIMOS", et, "J2000", "NONE", "MARS")
    return {
        "phobos": np.asarray(phobos_vec, dtype=float) * VISUAL_SCALE,
        "deimos": np.asarray(deimos_vec, dtype=float) * VISUAL_SCALE,
    }


def build_mars_system_figure(when_utc: datetime, slider_hour: int) -> go.Figure:
    positions = _mars_moon_positions(when_utc)
    mars_radius = MARS_RADIUS_KM * VISUAL_SCALE
    stars = _starfield()

    fig_3d = go.Figure()
    fig_3d.add_trace(_sphere_surface((0.0, 0.0, 0.0), mars_radius, "#CC5C2E", "Mars"))
    fig_3d.add_trace(
        go.Scatter3d(
            x=stars["x"],
            y=stars["y"],
            z=stars["z"],
            mode="markers",
            name="Stars",
            marker=dict(size=2, color="rgba(255,255,255,0.55)", opacity=0.55),
            hoverinfo="skip",
        )
    )
    fig_3d.add_trace(
        go.Scatter3d(
            x=[positions["phobos"][0]],
            y=[positions["phobos"][1]],
            z=[positions["phobos"][2]],
            mode="markers+text",
            name="Phobos",
            marker=dict(size=7, color="#A5A29E", symbol="circle"),
            text=["Phobos"],
            textposition="top center",
            textfont=dict(color="#D9D5CF", size=11, family="Courier New"),
            hovertemplate="Phobos<br>x %{x:.2f}<br>y %{y:.2f}<br>z %{z:.2f}<extra></extra>",
        )
    )
    fig_3d.add_trace(
        go.Scatter3d(
            x=[positions["deimos"][0]],
            y=[positions["deimos"][1]],
            z=[positions["deimos"][2]],
            mode="markers+text",
            name="Deimos",
            marker=dict(size=6, color="#8E857A", symbol="circle"),
            text=["Deimos"],
            textposition="top center",
            textfont=dict(color="#D9D5CF", size=10, family="Courier New"),
            hovertemplate="Deimos<br>x %{x:.2f}<br>y %{y:.2f}<br>z %{z:.2f}<extra></extra>",
        )
    )
    fig_3d.add_trace(
        go.Scatter3d(
            x=[0.0, positions["phobos"][0]],
            y=[0.0, positions["phobos"][1]],
            z=[0.0, positions["phobos"][2]],
            mode="lines",
            name="Phobos vector",
            line=dict(color="rgba(165,162,158,0.35)", width=2),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    limit = 35.0
    fig_3d.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=45, b=0),
        legend=dict(orientation="h", y=1.02, x=0.02),
        scene=dict(
            bgcolor="rgba(0,0,0,0)",
            xaxis=dict(visible=False, range=[-limit, limit], showbackground=False),
            yaxis=dict(visible=False, range=[-limit, limit], showbackground=False),
            zaxis=dict(visible=False, range=[-limit, limit], showbackground=False),
            aspectmode="data",
            camera=dict(eye=dict(x=1.8, y=1.5, z=0.9), center=dict(x=0.0, y=0.0, z=-0.05)),
        ),
        showlegend=False,
    )
    return fig_3d


def build_mars_system_animation_html(when_utc: datetime) -> str:
    frame_times = [when_utc + timedelta(seconds=offset) for offset in range(-180, 181, 15)]
    frames = [go.Frame(data=build_mars_system_figure(frame_time, frame_time.hour).data, name=frame_time.isoformat()) for frame_time in frame_times]
    animated_figure = build_mars_system_figure(when_utc, when_utc.hour)
    animated_figure.frames = frames
    return pio.to_html(
        animated_figure,
        full_html=False,
        include_plotlyjs="cdn",
        auto_play=True,
        config={"displayModeBar": True, "responsive": True},
    )

with st.expander("3D Mars / Phobos system view", expanded=True):
    visual_when = datetime(MISSION_DAY.year, MISSION_DAY.month, MISSION_DAY.day, hour, 0, 0, tzinfo=timezone.utc)
    st.markdown(
        f"<div style='margin:0.15rem 0 0.4rem;font-family:IBM Plex Mono, ui-monospace, monospace;font-size:0.92rem;letter-spacing:0.08em;color:#F5E6D3;'>Mars system visualized from SPICE at hour {hour:02d}:00 UTC</div>",
        unsafe_allow_html=True,
    )
    st.caption("SPICE-driven J2000 positions synced to the hour slider. Mars is intentionally enlarged for presentation clarity. The preview auto-plays a short, slow SPICE window around the current moment.")
    if live_mode:
        components.html(build_mars_system_animation_html(current_time), height=650, scrolling=False)
    else:
        st.plotly_chart(build_mars_system_figure(visual_when, hour), width="stretch")


@st.cache_data(show_spinner="Computing 24h SPICE / pvlib timeline…")
def mission_day_series() -> pd.DataFrame:
    start = datetime(MISSION_DAY.year, MISSION_DAY.month, MISSION_DAY.day, tzinfo=timezone.utc)
    times = [start + timedelta(minutes=10 * i) for i in range(24 * 6)]
    peak = TRANSIT_PEAK_UTC
    times += [peak + timedelta(seconds=s) for s in range(-50, 51, 2)]
    times = sorted(set(times))

    rows = []
    for t in times:
        transiting, blocked = get_transit_info(t)
        clear = expected_solar_output_pct(t)
        rows.append(
            {
                "utc": t,
                "hour": t.hour + t.minute / 60.0 + t.second / 3600.0,
                "solar_clear": clear,
                "solar_with_phobos": clear * (1.0 - blocked / 100.0),
                "blocked": blocked,
                "transiting": transiting,
            }
        )
    return pd.DataFrame(rows)
df = mission_day_series()

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=df["hour"],
        y=df["solar_clear"],
        name="Clear-sky baseline",
        line=dict(color="#8A6A4F", width=1, dash="dot"),
        hovertemplate="h %{x:.2f}<br>baseline %{y:.1f}%<extra></extra>",
    )
)
fig.add_trace(
    go.Scatter(
        x=df["hour"],
        y=df["solar_with_phobos"],
        name="Array output (Phobos)",
        line=dict(color=AMBER, width=2.4),
        hovertemplate="h %{x:.2f}<br>output %{y:.1f}%<extra></extra>",
    )
)
peak_hour = TRANSIT_PEAK_UTC.hour + TRANSIT_PEAK_UTC.minute / 60.0 + TRANSIT_PEAK_UTC.second / 3600.0
fig.add_vline(x=peak_hour, line_width=1, line_dash="dash", line_color="#FF4B4B")
fig.add_annotation(
    x=peak_hour,
    y=95,
    text="PHOBOS TRANSIT",
    showarrow=False,
    font=dict(color="#FF4B4B", family="Courier New", size=12),
    xanchor="left",
    xshift=8,
)
fig.add_vline(x=when.hour + when.minute / 60.0 + when.second / 3600.0, line_width=1, line_color="#F5E6D3")
peak_idx = int(df["blocked"].idxmax())
fig.add_trace(
    go.Scatter(
        x=[df.loc[peak_idx, "hour"]],
        y=[df.loc[peak_idx, "solar_with_phobos"]],
        mode="markers+text",
        name="Transit peak",
        marker=dict(size=14, color="#FF4B4B", symbol="x"),
        text=["DIP"],
        textposition="bottom center",
        textfont=dict(color="#FF4B4B", size=12),
    )
)
fig.add_trace(
    go.Scatter(
        x=df["hour"],
        y=df["blocked"],
        name="Phobos occultation %",
        line=dict(color="#FF4B4B", width=1.6),
        yaxis="y2",
        hovertemplate="h %{x:.3f}<br>blocked %{y:.1f}%<extra></extra>",
    )
)
fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(11,11,12,0.35)",
    font=dict(color="#F5E6D3", family="Courier New"),
    margin=dict(l=40, r=50, t=30, b=40),
    legend=dict(orientation="h", y=1.14),
    xaxis_title="Hour of day (UTC)",
    yaxis_title="Solar output (%)",
    yaxis=dict(range=[-2, 105]),
    yaxis2=dict(title="Disk blocked (%)", overlaying="y", side="right", range=[0, 40], showgrid=False, color="#FF4B4B"),
    xaxis=dict(range=[0, 24], dtick=2),
    height=360,
)

zoom = df[(df["hour"] >= peak_hour - 0.05) & (df["hour"] <= peak_hour + 0.05)]
st.markdown("#### Phobos transit — real SPICE geometry (~30 s)")
fig_z = go.Figure()
fig_z.add_trace(
    go.Scatter(
        x=zoom["utc"],
        y=zoom["solar_with_phobos"],
        name="Array output",
        line=dict(color=AMBER, width=3),
        fill="tozeroy",
        fillcolor="rgba(255,140,50,0.12)",
    )
)
fig_z.add_trace(
    go.Scatter(
        x=zoom["utc"],
        y=zoom["solar_clear"],
        name="Clear-sky baseline",
        line=dict(color="#8A6A4F", width=1.5, dash="dot"),
    )
)
fig_z.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(11,11,12,0.35)",
    font=dict(color="#F5E6D3", family="Courier New"),
    height=300,
    margin=dict(l=40, r=10, t=42, b=48),
    yaxis_title="Solar output (%)",
    legend=dict(orientation="h", y=1.15, x=0.0),
)
fig_z.update_xaxes(tickformat="%H:%M:%S", title_text="UTC time")
st.plotly_chart(fig_z, width="stretch")

st.markdown("#### Sol irradiance (24 h UTC)")
st.plotly_chart(fig, width="stretch")
st.caption("Red series (right axis) is disk occultation. The physical dip lasts ~30 seconds — judges should look at the zoom plot.")

st.markdown("#### Decision stack")
st.markdown(
    """
1. **SPICE** — when Phobos transits, and how much of the disk it covers  
2. **pvlib baseline** — what the array would produce without the moon  
3. **RF regressor** — settlement demand (hour + greenhouse)  
4. **RF classifier** — the one operational call: *do we curtail or go to batteries?*
    """
)
st.markdown(
    f"**Now:** blocked `{percent_blocked:.1f}%`  ·  demand `{demand_kw:.1f} kW`  ·  "
    f"call **{RISK_LABELS[risk_level]}**"
)
if risk_level == 0:
    action = "Hold nominal ops. Log irradiance."
elif risk_level == 1:
    action = "Dim habitat lights. Defer greenhouse grow-lights / ISRU heaters."
else:
    action = "Close battery contactors. Ride through on stored energy until egress."
st.success(f"Recommended action: {action}")
