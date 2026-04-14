"""
app.py
------
NutriTrack AI — Streamlit frontend.

Main page (original layout, single scroll):
  1. Log a Meal     — text description or photo upload for macro analysis
  2. Today's Log    — daily totals with RDI progress bars (persisted via SQLite)
  3. Suggestions    — gap-closing food recommendations

Sidebar:
  Ask about meals  — multi-day selector + AI chat grounded in your food log

Run with:
    python -m streamlit run app.py
"""

import uuid
import json
from datetime import datetime, date
import os

from dotenv import load_dotenv
load_dotenv()

import streamlit as st

from pipeline import analyze_meal, analyze_meal_from_image, RDI
from suggestions import get_suggestions
from parser import groq_request
import chat_history as db

GROQ_CHAT_MODEL = "llama-3.3-70b-versatile"

# ── app config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="NutriTrack AI",
    page_icon="🥗",
    layout="centered",
)

db.init_db()

# ── session state defaults ────────────────────────────────────────────────────

if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "last_mode" not in st.session_state:
    st.session_state.last_mode = "text"
if "last_label" not in st.session_state:
    st.session_state.last_label = ""
if "history_selected_dates" not in st.session_state:
    st.session_state.history_selected_dates = []
if "history_chat_messages" not in st.session_state:
    st.session_state.history_chat_messages = []
if "history_chat_session_id" not in st.session_state:
    st.session_state.history_chat_session_id = None
if "history_prev_dates" not in st.session_state:
    st.session_state.history_prev_dates = []
if "selected_date" not in st.session_state:
    st.session_state.selected_date = date.today()
if "last_synced_date" not in st.session_state:
    st.session_state.last_synced_date = None

# ── constants ─────────────────────────────────────────────────────────────────

TODAY = date.today().isoformat()

MACRO_COLORS = {
    "calories": "#f97316",
    "protein":  "#3b82f6",
    "carbs":    "#a855f7",
    "fat":      "#eab308",
}
MACRO_UNITS = {
    "calories": "kcal",
    "protein":  "g",
    "carbs":    "g",
    "fat":      "g",
}

# ── helpers ───────────────────────────────────────────────────────────────────

def get_daily_totals(log_date: str = TODAY) -> dict:
    totals = {"calories": 0.0, "protein": 0.0, "carbs": 0.0, "fat": 0.0}
    for entry in db.get_meals_for_date(log_date):
        totals["calories"] += entry["calories"] or 0
        totals["protein"]  += entry["protein"]  or 0
        totals["carbs"]    += entry["carbs"]     or 0
        totals["fat"]      += entry["fat"]       or 0
    totals["calories"] = round(totals["calories"])
    totals["protein"]  = round(totals["protein"]  * 10) / 10
    totals["carbs"]    = round(totals["carbs"]     * 10) / 10
    totals["fat"]      = round(totals["fat"]       * 10) / 10
    return totals


def get_daily_rdi(log_date: str = TODAY) -> dict:
    totals = db.get_daily_totals_db(log_date)
    return {
        macro: {
            "consumed":  totals.get(macro, 0) or 0,
            "target":    target,
            "percent":   round(((totals.get(macro, 0) or 0) / target) * 100),
            "remaining": max(0.0, round((target - (totals.get(macro, 0) or 0)) * 10) / 10),
        }
        for macro, target in RDI.items()
    }


def render_macro_cards(rdi: dict):
    cols = st.columns(4)
    for col, (macro, vals) in zip(cols, rdi.items()):
        unit  = MACRO_UNITS[macro]
        pct   = min(vals["percent"], 100)
        over  = vals["percent"] > 100
        color = "#ef4444" if over else MACRO_COLORS[macro]
        with col:
            st.markdown(
                f"""
                <div style="background:#fafafa;border:1px solid #e8e8e8;
                            border-radius:10px;padding:12px 14px;">
                  <div style="font-size:11px;color:#999;font-weight:500;
                              text-transform:capitalize;margin-bottom:4px;">{macro}</div>
                  <div style="font-size:20px;font-weight:700;color:#1a1a1a;
                              margin-bottom:2px;">{vals['consumed']}{unit}</div>
                  <div style="font-size:11px;color:{'#ef4444' if over else '#999'};
                              margin-bottom:8px;">{vals['percent']}% of {vals['target']}{unit}</div>
                  <div style="height:4px;background:#e8e8e8;border-radius:4px;">
                    <div style="width:{pct}%;height:100%;background:{color};
                                border-radius:4px;"></div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def ensure_history_session(selected_dates: list[str]) -> str:
    dates_key = ",".join(sorted(selected_dates))
    prev_key  = ",".join(sorted(st.session_state.history_prev_dates))
    if dates_key != prev_key or st.session_state.history_chat_session_id is None:
        sid = str(uuid.uuid4())[:12]
        title = "Chat: " + ", ".join(
            datetime.strptime(d, "%Y-%m-%d").strftime("%b %d")
            for d in sorted(selected_dates)
        )
        db.create_session(sid, title)
        st.session_state.history_chat_session_id = sid
        st.session_state.history_chat_messages   = []
        st.session_state.history_prev_dates      = selected_dates
    return st.session_state.history_chat_session_id


def build_day_context(selected_dates: list[str]) -> str:
    lines = []
    for d in sorted(selected_dates):
        label  = datetime.strptime(d, "%Y-%m-%d").strftime("%A %B %d, %Y")
        meals  = db.get_meals_for_date(d)
        totals = db.get_daily_totals_db(d)
        lines.append(f"=== {label} ===")
        if not meals:
            lines.append("  No meals logged.")
        else:
            for m in meals:
                lines.append(
                    f"  [{m['meal_type'].capitalize()}] {m['label']} — "
                    f"{m['calories']:.0f} kcal | "
                    f"{m['protein']}g protein | "
                    f"{m['carbs']}g carbs | "
                    f"{m['fat']}g fat"
                )
            lines.append(
                f"  DAILY TOTAL: {totals['calories']:.0f} kcal | "
                f"{totals['protein']}g protein | "
                f"{totals['carbs']}g carbs | "
                f"{totals['fat']}g fat"
            )
        lines.append("")
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Ask about meals
# ════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 💬 Ask about meals")

    logged_dates = db.get_logged_dates()

    if not logged_dates:
        st.info("Log some meals first to ask questions about them.")
    else:
        # Sync sidebar selection when the main-page date picker changes
        _sel_iso = st.session_state.selected_date.isoformat()
        if _sel_iso != st.session_state.last_synced_date:
            st.session_state.last_synced_date = _sel_iso
            if _sel_iso in logged_dates:
                # Put the newly selected date first, keep any other already-chosen dates
                _others = [d for d in st.session_state.history_selected_dates if d != _sel_iso]
                st.session_state.history_selected_dates = [_sel_iso] + _others

        # Fallback: if nothing selected yet, default to first logged date
        _sidebar_default = st.session_state.history_selected_dates or logged_dates[:1]
        # Guard: remove any stale dates no longer in logged_dates
        _sidebar_default = [d for d in _sidebar_default if d in logged_dates] or logged_dates[:1]

        selected_dates = st.multiselect(
            "Select days to chat about",
            options=logged_dates,
            default=_sidebar_default,
            format_func=lambda d: datetime.strptime(d, "%Y-%m-%d").strftime("%a, %b %d %Y"),
        )
        st.session_state.history_selected_dates = selected_dates

        if not selected_dates:
            st.warning("Select at least one day above.")
        else:
            session_id  = ensure_history_session(selected_dates)
            day_context = build_day_context(selected_dates)

            system_prompt = (
                "You are a knowledgeable nutrition assistant for NutriTrack AI.\n"
                "The user has selected specific logged days. Use the meal data below "
                "as the factual basis for all answers. Be concise, friendly, and precise.\n\n"
                "LOGGED MEAL DATA:\n"
                + day_context
                + "\nRDI targets: Calories 2000 kcal · Protein 50 g · Carbs 275 g · Fat 78 g."
            )

            # Chat history
            for msg in st.session_state.history_chat_messages:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])

            # Starter suggestions (shown only before first message)
            if not st.session_state.history_chat_messages:
                date_str = ", ".join(
                    datetime.strptime(d, "%Y-%m-%d").strftime("%b %d")
                    for d in sorted(selected_dates)
                )
                st.caption(f"Try asking about {date_str}:")
                starters = [
                    "How did my protein compare to the RDI?",
                    "Which meal had the most calories?",
                    "What nutrients am I most deficient in?",
                    "Compare my intake across the selected days.",
                ]
                for i, s in enumerate(starters):
                    if st.button(s, key=f"starter_{i}", use_container_width=True):
                        with st.chat_message("user"):
                            st.write(s)
                        db.add_message(session_id, "user", s)
                        st.session_state.history_chat_messages.append({"role": "user", "content": s})
                        with st.chat_message("assistant"):
                            with st.spinner("Thinking…"):
                                try:
                                    resp  = groq_request({
                                        "model": GROQ_CHAT_MODEL,
                                        "max_tokens": 600,
                                        "messages": [
                                            {"role": "system", "content": system_prompt},
                                            *[{"role": m["role"], "content": m["content"]}
                                              for m in st.session_state.history_chat_messages],
                                        ],
                                    })
                                    reply = resp["choices"][0]["message"]["content"]
                                except Exception as e:
                                    reply = f"⚠️ {e}"
                                st.write(reply)
                        db.add_message(session_id, "assistant", reply)
                        st.session_state.history_chat_messages.append({"role": "assistant", "content": reply})
                        db.update_session_meta(session_id, title=s[:40], preview=reply[:80])
                        st.rerun()

            if prompt := st.chat_input("Ask about your meals…"):
                with st.chat_message("user"):
                    st.write(prompt)
                db.add_message(session_id, "user", prompt)
                st.session_state.history_chat_messages.append({"role": "user", "content": prompt})

                with st.chat_message("assistant"):
                    with st.spinner("Thinking…"):
                        try:
                            resp  = groq_request({
                                "model": GROQ_CHAT_MODEL,
                                "max_tokens": 600,
                                "messages": [
                                    {"role": "system", "content": system_prompt},
                                    *[{"role": m["role"], "content": m["content"]}
                                      for m in st.session_state.history_chat_messages],
                                ],
                            })
                            reply = resp["choices"][0]["message"]["content"]
                        except Exception as e:
                            reply = f"⚠️ {e}"
                        st.write(reply)

                db.add_message(session_id, "assistant", reply)
                st.session_state.history_chat_messages.append({"role": "assistant", "content": reply})
                if len(st.session_state.history_chat_messages) == 2:
                    db.update_session_meta(
                        session_id,
                        title=prompt[:40] + ("…" if len(prompt) > 40 else ""),
                        preview=reply[:80],
                    )
                st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# MAIN PAGE — original single-scroll layout
# ════════════════════════════════════════════════════════════════════════════

st.markdown(
    """
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
      <div style="width:36px;height:36px;border-radius:8px;
                  background:linear-gradient(135deg,#22c55e,#16a34a);
                  display:flex;align-items:center;justify-content:center;
                  font-size:18px;">🥗</div>
      <span style="font-weight:700;font-size:22px;letter-spacing:-0.3px;">NutriTrack AI</span>
    </div>
    <p style="color:#888;font-size:13px;margin-bottom:24px;">{}</p>
    """.format(datetime.now().strftime("%A, %B %d, %Y").replace(" 0", " ")),
    unsafe_allow_html=True,
)

# ── Date picker ─────────────────────────────────────────────────────────────

_today      = date.today()
_month_start = date(_today.year, _today.month, 1)

_col_title, _col_date = st.columns([3, 2])
with _col_date:
    st.session_state.selected_date = st.date_input(
        "Viewing date",
        value=st.session_state.selected_date,
        min_value=_month_start,
        max_value=_today,
        format="DD/MM/YYYY",
        label_visibility="collapsed",
    )

SEL_DATE = st.session_state.selected_date.isoformat()
is_today  = (SEL_DATE == TODAY)

# ── Log a Meal ────────────────────────────────────────────────────────────────

st.markdown("### Log a Meal")

mode = st.radio(
    "Input mode",
    options=[" Text", " Photo"],
    horizontal=True,
    label_visibility="collapsed",
)

if mode == " Text":
    meal_text = st.text_input(
        "Meal description",
        placeholder="e.g. 2 scrambled eggs with toast and orange juice",
        label_visibility="collapsed",
    )
    if st.button("Analyze", type="primary", disabled=not meal_text.strip()):
        with st.spinner("Analyzing meal…"):
            try:
                result = analyze_meal(meal_text.strip())
                st.session_state.last_result = result
                st.session_state.last_mode   = "text"
                st.session_state.last_label  = meal_text.strip()
            except Exception as e:
                st.error(f"⚠️ {e}")
else:
    uploaded = st.file_uploader(
        "Upload a meal photo",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )
    if uploaded:
        st.image(uploaded, width=320)
        if st.button("Analyze Photo", type="primary"):
            with st.spinner("Analyzing photo…"):
                try:
                    image_bytes = uploaded.read()
                    mime_type   = uploaded.type or "image/jpeg"
                    result = analyze_meal_from_image(image_bytes, mime_type)
                    st.session_state.last_result = result
                    st.session_state.last_mode   = "photo"
                    st.session_state.last_label  = f"Photo: {uploaded.name}"
                except Exception as e:
                    st.error(f"⚠️ {e}")

# ── Analysis result ───────────────────────────────────────────────────────────

if st.session_state.last_result:
    result = st.session_state.last_result
    st.markdown("---")
    render_macro_cards(result["rdi"])

    with st.expander("View ingredient breakdown"):
        rows = []
        for f in result["food_results"]:
            if f["found"]:
                rows.append({
                    "Food":    f["name"],
                    "Qty":     f"{f['quantity']} {f['unit']}",
                    "~g":      f["estimated_grams"],
                    "kcal":    f["scaled"]["calories"],
                    "Protein": f"{f['scaled']['protein']}g",
                    "Carbs":   f"{f['scaled']['carbs']}g",
                    "Fat":     f"{f['scaled']['fat']}g",
                    "Matched": f["matched_description"],
                })
            else:
                rows.append({
                    "Food":    f["name"],
                    "Qty":     f"{f['quantity']} {f['unit']}",
                    "~g":      "—",
                    "kcal":    "Not found",
                    "Protein": "", "Carbs": "", "Fat": "", "Matched": "",
                })
        if rows:
            st.dataframe(rows, use_container_width=True)

    if st.session_state.last_mode == "photo":
        st.info("📊 Photo estimates are approximate. For precise tracking, use text input.")

    meal_type = st.selectbox(
        "Meal type",
        ["breakfast", "lunch", "dinner", "snack", "meal"],
        index=4,
    )

    if st.button("＋ Log this meal", type="primary"):
        db.log_meal(
            label=st.session_state.last_label,
            food_results=result["food_results"],
            totals=result["totals"],
            meal_type=meal_type,
            log_date=SEL_DATE,
        )
        st.session_state.last_result = None
        st.success(f"✓ Logged to {st.session_state.selected_date.strftime('%b %d')}!")
        st.rerun()

st.markdown("---")

# ── Daily Progress ───────────────────────────────────────────────────────────

_progress_label = "Today's Progress" if is_today else st.session_state.selected_date.strftime("%B %d — Meal Log")

header_cols = st.columns([5, 1])
with header_cols[0]:
    st.markdown(f"### {_progress_label}")
with header_cols[1]:
    meals_today = db.get_meals_for_date(SEL_DATE)
    if meals_today:
        if st.button("Clear", type="secondary"):
            for entry in meals_today:
                db.delete_meal_entry(entry["id"])
            st.rerun()

meals_today = db.get_meals_for_date(SEL_DATE)

if not meals_today:
    _empty_msg = "No meals logged today. Analyze a meal above and hit \"Log this meal\"." if is_today \
        else f"No meals logged on {st.session_state.selected_date.strftime('%B %d')}."
    st.markdown(
        f"<div style='text-align:center;padding:2rem 0;color:#bbb;font-size:14px;'>{_empty_msg}</div>",
        unsafe_allow_html=True,
    )
else:
    daily_rdi = get_daily_rdi(SEL_DATE)
    render_macro_cards(daily_rdi)

    st.markdown("<br>", unsafe_allow_html=True)

    for macro, vals in daily_rdi.items():
        unit = MACRO_UNITS[macro]
        st.caption(f"{macro.capitalize()}: {vals['consumed']}{unit} / {vals['target']}{unit} ({vals['percent']}%)")
        st.progress(min(vals["percent"] / 100, 1.0))

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Meal Log**")

    for entry in meals_today:
        with st.container():
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"**{entry['label']}**")
                st.markdown(
                    f"<span style='color:#f97316'>{entry['calories']:.0f} kcal</span> &nbsp;"
                    f"<span style='color:#3b82f6'>{entry['protein']}g protein</span> &nbsp;"
                    f"<span style='color:#a855f7'>{entry['carbs']}g carbs</span> &nbsp;"
                    f"<span style='color:#eab308'>{entry['fat']}g fat</span>",
                    unsafe_allow_html=True,
                )
            with col2:
                st.caption(
                    datetime.fromisoformat(entry["logged_at"]).strftime("%I:%M %p")
                )
            st.markdown("---")

st.markdown("<br>", unsafe_allow_html=True)

# ── Suggestions ───────────────────────────────────────────────────────────────

if meals_today:
    st.markdown("### Suggestions")

    if st.button("Suggest foods"):
        with st.spinner("Finding suggestions…"):
            try:
                daily_rdi  = get_daily_rdi(SEL_DATE)
                suggestion = get_suggestions(daily_rdi)

                if suggestion["all_met"]:
                    st.success("🎉 All daily targets met — great job today!")
                else:
                    if suggestion["deficient"]:
                        st.markdown(
                            f"Most deficient in: **{', '.join(suggestion['deficient'][:2])}**. "
                            "Here are some foods to help close the gap:"
                        )
                    for s in suggestion["suggestions"]:
                        with st.container():
                            st.markdown(f"**{s['description']}**")
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("Calories", f"{s['calories']} kcal")
                            col2.metric("Protein",  f"{s['protein']}g")
                            col3.metric("Carbs",    f"{s['carbs']}g")
                            col4.metric("Fat",      f"{s['fat']}g")
                            st.caption("per 100g")
                            st.markdown("---")
            except Exception as e:
                st.error(f"⚠️ {e}")
