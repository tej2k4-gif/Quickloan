# app.py — QuickLoan Risk Dashboard  (Stretch goal, upgraded)
# Run locally:  streamlit run app.py
# Needs (same folder):  loan_model.joblib  and  loans.csv
# The notebook saves the model like this (threshold is optional — without it the 50 % rule is used):
#   joblib.dump({'model': clf, 'columns': list(X_train.columns), 'threshold': 0.14, 'name': 'Stacking'}, 'loan_model.joblib')
import streamlit as st, pandas as pd, numpy as np, joblib
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="QuickLoan Risk Dashboard", page_icon="💰", layout="wide")

# ── Look & feel ──────────────────────────────────────────────────────────────
FAMILY = {"rf": "#059669", "gb": "#F472B6", "xgb": "#F472B6", "lr": "#38BDF8"}   # bagging / boosting / linear
GOOD, WARN, BAD, VIOLET, AMBER = "#0ca30c", "#ec835a", "#d03b3b", "#7C3AED", "#F59E0B"
st.markdown("""
<style>
.block-container {padding-top: 3.2rem; max-width: 1250px;}
.hero {background: linear-gradient(120deg, #7C3AED 0%, #DB2777 55%, #F59E0B 100%); color: #fff;
       padding: 22px 28px; border-radius: 18px; margin-bottom: 18px; box-shadow: 0 10px 30px rgba(124,58,237,.25);}
.hero h1 {color: #fff; font-size: 2.0rem; margin: 0 0 4px 0; padding: 0;}
.hero p {margin: 0; opacity: .95; font-size: 1.02rem;}
.verdict {border-radius: 16px; padding: 18px 22px; color: #fff; margin: 4px 0 10px 0;}
.verdict .big {font-size: 2.1rem; font-weight: 800; letter-spacing: .5px;}
.verdict .small {font-size: 1rem; opacity: .95;}
.approve {background: linear-gradient(120deg, #059669, #10b981);}
.reject  {background: linear-gradient(120deg, #be123c, #f43f5e);}
.chip {display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: .82rem; font-weight: 600;
       background: rgba(124,58,237,.12); color: #7C3AED; margin-right: 6px;}
div[data-testid="stMetric"] {background: rgba(124,58,237,.06); border: 1px solid rgba(124,58,237,.18);
       border-radius: 14px; padding: 12px 16px;}
</style>""", unsafe_allow_html=True)

try:  # Streamlit ≥ 1.50 prefers width="stretch"; older versions use use_container_width
    _v = tuple(int(x) for x in st.__version__.split(".")[:2])
except ValueError:
    _v = (1, 0)
STRETCH = {"width": "stretch"} if _v >= (1, 50) else {"use_container_width": True}

def show(fig, h=340, margin=None):
    fig.update_layout(height=h, margin=margin or dict(l=10, r=20, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(size=13))
    st.plotly_chart(fig, config={"displayModeBar": False}, **STRETCH)

def aed(x):
    return ("−" if x < 0 else "") + f"AED {abs(x):,.0f}"

# ── Model & data ─────────────────────────────────────────────────────────────
@st.cache_resource
def load():
    M = joblib.load("loan_model.joblib")
    return M["model"], M["columns"], float(M.get("threshold", 0.5)), M.get("name", type(M["model"]).__name__)
model, COLUMNS, TUNED_T, MODEL_NAME = load()

def encode(df):
    return pd.get_dummies(df, columns=["home_ownership"], dtype=int).reindex(columns=COLUMNS, fill_value=0)

def money_saved(y_true, y_pred, loan, interest=0.20):
    y_true, y_pred, loan = np.array(y_true), np.array(y_pred), np.array(loan)
    caught = (y_pred==1)&(y_true==1)        # defaulter rejected -> loan saved
    wrong  = (y_pred==1)&(y_true==0)        # good borrower rejected -> interest lost
    return loan[caught].sum() - interest*loan[wrong].sum()

@st.cache_resource
def test_set():
    df = pd.read_csv("loans.csv")
    X = encode(df.drop(columns="default")); y = df["default"]
    Xtr,Xte,ytr,yte,lo_tr,lo_te = train_test_split(X,y,df["loan_amount"],test_size=0.25,random_state=1,stratify=y)
    return yte.values, lo_te.values, model.predict_proba(Xte)[:, 1]

# ── Sidebar controls ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎛️ Decision settings")
    interest = st.slider("Interest earned on a good loan", 0.05, 0.50, 0.20, 0.01, format="%.2f",
                         help="What QuickLoan loses when it rejects a borrower who would have repaid.")
    break_even = interest / (1 + interest)
    threshold = st.slider("Reject when P(default) ≥", 0.02, 0.90, round(TUNED_T, 2), 0.01,
                          help=f"Tuned on training data: {TUNED_T:.2f}. Break-even at this interest rate: {break_even:.2f}.")
    st.caption(f"🔒 Tuned cut-off **{TUNED_T:.2f}** · ⚖️ break-even **{break_even:.2f}** · 🐢 brief default **0.50**")
    st.markdown("---")
    st.markdown(f"**Deployed model:** {MODEL_NAME}")
    st.markdown("**Payoff per loan**\n\n| | approve | reject |\n|---|---|---|\n"
                f"| repays | +{interest:.0%} | 0 |\n| defaults | −100 % | 0 ✅ |")

st.markdown(f"""<div class="hero"><h1>💰 QuickLoan · Risk Dashboard</h1>
<p>{MODEL_NAME} ensemble · rejects when default risk ≥ <b>{threshold:.0%}</b> · interest {interest:.0%}.
Every number below is in dirhams, not accuracy.</p></div>""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🧾 1 · Loan decision", "💵 2 · Money saved (portfolio)", "🎚️ 3 · Threshold lab"])

# ---------- CARD 1: Decision ----------
with tab1:
    c1, c2, c3 = st.columns([1, 1, 1.25])
    with c1:
        age = st.slider("Age", 21, 64, 35)
        annual_income = st.number_input("Annual income (AED)", 24000, 400000, 90000, step=1000)
        loan_amount = st.number_input("Loan amount (AED)", 5000, 150000, 40000, step=1000)
        loan_term_months = st.selectbox("Loan term (months)", [12,24,36,48,60], index=2)
        employment_years = st.slider("Employment years", 0.0, 35.0, 4.0, 0.5)
    with c2:
        credit_history_months = st.slider("Credit history (months)", 0, 300, 60)
        num_existing_loans = st.slider("Existing loans", 0, 8, 1)
        past_defaults = st.slider("Past defaults", 0, 6, 0)
        home_ownership = st.selectbox("Home ownership", ["rent","own","mortgage"])
        debt_to_income = round(loan_amount/(annual_income+1), 2)
        st.markdown(f'<span class="chip">Debt-to-income (auto): {debt_to_income}</span>', unsafe_allow_html=True)

    applicant = dict(age=age, annual_income=annual_income, loan_amount=loan_amount,
        loan_term_months=loan_term_months, employment_years=employment_years,
        credit_history_months=credit_history_months, num_existing_loans=num_existing_loans,
        past_defaults=past_defaults, debt_to_income=debt_to_income, home_ownership=home_ownership)
    X1 = encode(pd.DataFrame([applicant]))
    prob = float(model.predict_proba(X1)[0][1])
    pred = int(prob >= threshold)
    ev_approve = (1 - prob) * interest * loan_amount - prob * loan_amount   # expected profit if we approve

    with c3:
        if pred == 1:
            st.markdown(f'<div class="verdict reject"><div class="big">❌ REJECT</div>'
                        f'<div class="small">Default risk {prob:.0%} ≥ cut-off {threshold:.0%}</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="verdict approve"><div class="big">✅ APPROVE</div>'
                        f'<div class="small">Default risk {prob:.0%} &lt; cut-off {threshold:.0%}</div></div>', unsafe_allow_html=True)
        g = go.Figure(go.Indicator(
            mode="gauge+number", value=prob * 100, number={"suffix": "%", "font": {"size": 40}},
            title={"text": "Predicted default risk"},
            gauge={"axis": {"range": [0, 100], "ticksuffix": "%"},
                   "bar": {"color": "#1d1b2e", "thickness": 0.25},
                   "steps": [{"range": [0, threshold * 100], "color": "rgba(16,185,129,.35)"},
                             {"range": [threshold * 100, 100], "color": "rgba(244,63,94,.35)"}],
                   "threshold": {"line": {"color": VIOLET, "width": 4}, "thickness": 0.9, "value": threshold * 100}}))
        show(g, 260, margin=dict(l=40, r=50, t=50, b=10))
        st.metric("Expected profit if approved", aed(ev_approve),
                  help="(1 − p) × interest × loan − p × loan. Rejecting is worth it when this is negative.")

    # Stacking: show what each expert said and how the manager combined them
    if hasattr(model, "named_estimators_") and hasattr(model, "final_estimator_"):
        st.markdown("#### 🧑‍💼 Inside the committee: what each expert says about this applicant")
        names = list(model.named_estimators_.keys())
        probs = [float(model.named_estimators_[n].predict_proba(X1)[0][1]) for n in names]
        labels = {"rf": "Random Forest", "gb": "Gradient Boosting", "xgb": "XGBoost", "lr": "Logistic Reg."}
        ylab = [labels.get(n, n) for n in names] + ["➜ Manager (final)"]
        cols = [FAMILY.get(n, "#94a3b8") for n in names] + [AMBER]
        f = go.Figure(go.Bar(x=[p * 100 for p in probs] + [prob * 100], y=ylab, orientation="h", marker_color=cols,
                             text=[f"{p:.0%}" for p in probs + [prob]], textposition="outside", cliponaxis=False))
        f.add_vline(x=threshold * 100, line_color=VIOLET, line_width=2,
                    annotation_text=f"cut-off {threshold:.0%}", annotation_font_color=VIOLET)
        f.update_xaxes(range=[0, max(probs + [prob, threshold]) * 100 * 1.25 + 2], ticksuffix="%")
        f.update_yaxes(autorange="reversed")
        show(f, 280)
        w = model.final_estimator_.coef_[0]
        st.caption("Manager's trust weights: " + " · ".join(f"{labels.get(n, n)} {v:+.2f}" for n, v in zip(names, w)))

# ---------- CARD 2: Money saved on the portfolio ----------
with tab2:
    try:
        yte, lo_te, p_te = test_set()
        pred_te = (p_te >= threshold).astype(int)
        saved = money_saved(yte, pred_te, lo_te, interest)
        saved_50 = money_saved(yte, (p_te >= 0.5).astype(int), lo_te, interest)
        caught = int(((pred_te==1)&(yte==1)).sum()); total_def = int((yte==1).sum())
        wrong  = int(((pred_te==1)&(yte==0)).sum())
        acc = float((pred_te == yte).mean())
        m1,m2,m3,m4 = st.columns(4)
        d = saved - saved_50
        m1.metric("💵 Money saved vs approve-all", f"AED {saved/1e6:.2f} M" if saved >= 0 else f"−AED {abs(saved)/1e6:.2f} M",
                  delta=f"{'+' if d >= 0 else '-'}AED {abs(d)/1e6:.2f} M vs 50 % rule", help=f"Exactly {aed(saved)}")
        m2.metric("🎯 Defaulters caught", f"{caught}/{total_def}", delta=f"{caught/total_def:.0%} recall", delta_color="off")
        m3.metric("😕 Good borrowers rejected", f"{wrong}", delta=f"of {int((yte==0).sum())}", delta_color="off")
        m4.metric("📏 Accuracy", f"{acc:.1%}", delta="not the goal", delta_color="off")

        loans_saved = lo_te[(pred_te==1)&(yte==1)].sum()
        int_lost = interest * lo_te[(pred_te==1)&(yte==0)].sum()
        still_lost = lo_te[(pred_te==0)&(yte==1)].sum()
        b = go.Figure(go.Bar(
            y=["✅ Loans saved (defaulters caught)", "👎 Interest lost (good borrowers rejected)",
               "= Net money saved", "💸 Still lost (defaulters missed)"],
            x=[loans_saved, -int_lost, saved, -still_lost], orientation="h",
            marker_color=[GOOD, WARN, AMBER, BAD],
            text=[aed(v) for v in [loans_saved, -int_lost, saved, -still_lost]], textposition="outside", cliponaxis=False))
        b.update_yaxes(autorange="reversed")
        b.update_xaxes(tickprefix="AED ", range=[-max(still_lost, int_lost) * 1.45, max(loans_saved, saved) * 1.35])
        b.update_layout(title="Where the money comes from (1,500 held-out loans)")
        show(b, 330)
        st.caption("Evaluated on the same held-out 25 % test split used in the notebook. "
                   "💸 is the prize still on the table — better recall shrinks it.")
    except FileNotFoundError:
        st.info("Put loans.csv in this folder to see the money-saved view.")

# ---------- CARD 3: Threshold lab ----------
with tab3:
    try:
        yte, lo_te, p_te = test_set()
        grid = np.round(np.arange(0.02, 0.91, 0.01), 2)
        money = [money_saved(yte, (p_te >= t).astype(int), lo_te, interest) for t in grid]
        recall = [((p_te >= t) & (yte == 1)).sum() / (yte == 1).sum() for t in grid]
        f = go.Figure()
        f.add_trace(go.Scatter(x=grid, y=money, mode="lines", line=dict(color=AMBER, width=3), fill="tozeroy",
                               fillcolor="rgba(245,158,11,.12)", name="Money saved",
                               customdata=np.array(recall) * 100,
                               hovertemplate="cut-off %{x:.2f}<br>AED %{y:,.0f}<br>recall %{customdata:.0f}%<extra></extra>"))
        for x, lab, col, pos in [(threshold, f"you: {threshold:.2f}", VIOLET, "top right"),
                                 (break_even, f"break-even {break_even:.2f}", "#1d1b2e", "bottom right"),
                                 (0.5, "50 % rule", "#8a87a0", "top left")]:
            f.add_vline(x=x, line_color=col, line_width=2, annotation_text=lab, annotation_font_color=col,
                        annotation_position=pos)
        f.update_layout(title="Money saved on the test set as you move the cut-off", showlegend=False,
                        xaxis_title="Reject when P(default) ≥ cut-off", yaxis_tickprefix="AED ")
        show(f, 380)
        best = grid[int(np.argmax(money))]
        near = "close to" if abs(best - TUNED_T) <= 0.05 else "some way from"
        st.info(f"At {interest:.0%} interest the payoff table's break-even risk is **{break_even:.1%}** "
                f"(= interest ÷ (1 + interest)). The deployed cut-off **{TUNED_T:.2f}** was tuned on training data only "
                f"(at 20 % interest); the best cut-off *in hindsight* on this test set is **{best:.2f}**, {near} it. "
                f"If the interest rate changes, re-tune — the break-even value is the natural starting point.")
    except FileNotFoundError:
        st.info("Put loans.csv in this folder to see the threshold lab.")
