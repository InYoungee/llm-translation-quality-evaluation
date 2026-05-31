import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="LLM Translation Evaluation Dashboard",
    page_icon="🔍",
    layout="wide"
)

# ===== CUSTOM CSS =====
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; font-size: 15px; }
    h1, h2, h3 { font-family: 'Space Mono', monospace; }

    /* Enlarge tab labels */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 17px !important;
        font-weight: 600 !important;
        padding: 14px 24px !important;
        letter-spacing: 0.02em !important;
    }
    .stTabs [aria-selected="true"] {
        font-size: 17px !important;
        font-weight: 700 !important;
    }
    .stTabs [data-baseweb="tab"] p {
        font-size: 17px !important;
        font-weight: 600 !important;
    }
    .block-container { padding-left: 4rem !important; padding-right: 4rem !important; }
    .kpi-card {
        background: linear-gradient(135deg, #1e2130, #252a3d);
        border: 1px solid #2e3450;
        border-radius: 12px;
        padding: 22px 26px;
        text-align: center;
    }
    .kpi-label { font-size: 13px; letter-spacing: 0.08em; text-transform: uppercase;
                 color: #7b8ab8; font-family: 'Space Mono', monospace; margin-bottom: 8px; }
    .kpi-value { font-size: 34px; font-weight: 700; font-family: 'Space Mono', monospace; }
    .kpi-sub   { font-size: 13px; color: #7b8ab8; margin-top: 4px; }
    .kpi-purple { color: #a78bfa; }
    .kpi-green  { color: #4ade80; }
    .kpi-blue   { color: #60a5fa; }
    .kpi-yellow { color: #fbbf24; }
    .kpi-coral  { color: #fb923c; }
    .section-title {
        font-family: 'Space Mono', monospace; font-size: 16px;
        letter-spacing: 0.08em; text-transform: uppercase; color: #7b8ab8;
        border-bottom: 1px solid #2e3450; padding-bottom: 12px; margin-bottom: 22px;
    }
    .insight-box {
        background: linear-gradient(135deg, #1a1f35, #1e2440);
        border-left: 3px solid #60a5fa;
        border-radius: 0 8px 8px 0;
        padding: 16px 20px; font-size: 15px; color: #b0bcd8; margin-top: 12px; line-height: 1.6;
    }
    .warning-box {
        background: linear-gradient(135deg, #1a1f35, #1e2440);
        border-left: 3px solid #fbbf24;
        border-radius: 0 8px 8px 0;
        padding: 16px 20px; font-size: 15px; color: #b0bcd8; margin-top: 12px; line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# ===== LOAD DATA =====
@st.cache_data
def load_data():
    scores = pd.read_excel('scoring_sheet_1to5_result.xlsx')
    bleu_ter = pd.read_excel('bleu_ter_summary.xlsx')
    dims = ['Accuracy','Fluency','Cultural_Appropriateness','Terminology','Naturalness']
    scores['Total_25'] = scores[dims].sum(axis=1)
    return scores, bleu_ter, dims

try:
    df, bleu_df, dims = load_data()
except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()

# ===== COLORS =====
COLORS = {'Claude': '#a78bfa', 'GPT4o': '#4ade80'}
COND_COLORS = {'A': '#94A3B8', 'B': '#60a5fa', 'C': '#fbbf24'}
COND_LABELS = {'A': 'A: No Context', 'B': 'B: + Context', 'C': 'C: + Context + Glossary'}
BG = 'rgba(0,0,0,0)'
GRID = '#2e3450'
FONT = '#b0bcd8'

def base_layout(title, height=380):
    return dict(
        title=dict(text=title, font=dict(size=13, family='Space Mono', color=FONT)),
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(color=FONT),
        height=height,
        margin=dict(t=50, b=20, l=20, r=20)
    )

# ===== AGGREGATES =====
model_cond = df.groupby(['Model','Condition'])['Total_25'].mean().reset_index()
model_dim  = df.groupby('Model')[dims].mean().reset_index()
cond_dim   = df.groupby('Condition')[dims].mean().reset_index()
cond_means = df.groupby('Condition')['Total_25'].mean()
model_means= df.groupby('Model')['Total_25'].mean()

claude_str = df[df['Model']=='Claude'].groupby('ID')['Total_25'].mean()
gpt_str    = df[df['Model']=='GPT4o'].groupby('ID')['Total_25'].mean()
comp       = pd.DataFrame({'Claude': claude_str, 'GPT4o': gpt_str})

# ===== HEADER =====
st.markdown("# 🔍 LLM Translation Evaluation")
st.markdown("**Claude vs GPT-4o · KO→EN Drama Dialogue · Stranger S01E01 · 74 strings · 3 conditions**")
st.markdown("---")

# ===== TABS =====
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview",
    "📐 Dimensions",
    "🤖 Model Comparison",
    "📏 BLEU / TER"
])

# ===========================
# TAB 1: OVERVIEW
# ===========================
with tab1:
    st.markdown('<div class="section-title">Key Metrics</div>', unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">Claude Overall</div>
            <div class="kpi-value kpi-purple">{model_means['Claude']:.1f}<span style="font-size:14px;color:#7b8ab8">/25</span></div>
            <div class="kpi-sub">avg all conditions</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">GPT-4o Overall</div>
            <div class="kpi-value kpi-green">{model_means['GPT4o']:.1f}<span style="font-size:14px;color:#7b8ab8">/25</span></div>
            <div class="kpi-sub">avg all conditions</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        boost_b = cond_means['B'] - cond_means['A']
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">Context Boost</div>
            <div class="kpi-value kpi-blue">{boost_b:+.2f}</div>
            <div class="kpi-sub">A → B improvement</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        boost_c = cond_means['C'] - cond_means['B']
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">Glossary Boost</div>
            <div class="kpi-value kpi-yellow">{boost_c:+.2f}</div>
            <div class="kpi-sub">B → C improvement</div>
        </div>""", unsafe_allow_html=True)
    with col5:
        weakest = df[dims].mean().idxmin().replace('_',' ')
        weakest_score = df[dims].mean().min()
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">Weakest Dimension</div>
            <div class="kpi-value" style="font-size:16px;color:#f87171">{weakest}</div>
            <div class="kpi-sub">avg {weakest_score:.2f}/5</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Overall Score — Model × Condition</div>', unsafe_allow_html=True)

    fig1 = go.Figure()
    for model in ['Claude', 'GPT4o']:
        sub = model_cond[model_cond['Model'] == model]
        fig1.add_trace(go.Bar(
            name=model,
            x=[COND_LABELS[c] for c in sub['Condition']],
            y=sub['Total_25'].round(2),
            marker_color=COLORS[model],
            opacity=0.85,
            text=sub['Total_25'].round(2),
            textposition='outside',
            textfont=dict(size=12)
        ))
    fig1.add_hline(y=12.5, line_dash='dot', line_color='#f87171',
                   annotation_text='Midpoint 12.5',
                   annotation_font=dict(color='#f87171', size=10))
    fig1.update_layout(**base_layout('Average Total Score by Model and Condition'),
        barmode='group',
        yaxis=dict(range=[18, 23], title='Score /25', gridcolor=GRID),
        xaxis=dict(gridcolor=GRID),
        legend=dict(orientation='h', y=1.15, x=0.5, xanchor='center')
    )
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("""<div class="insight-box">
    ⚡ <b>Key Finding:</b> Claude (20.65) slightly edges GPT-4o (20.41) overall.
    GPT-4o benefits more from context (+1.39 A→B) while Claude is more stable across conditions.
    Both models peak at Condition C but the glossary improvement is modest, consistent with the finding
    that over-specified glossary entries can sometimes hurt rather than help.
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Score Heatmap</div>', unsafe_allow_html=True)

    pivot = df.groupby(['Model','Condition'])['Total_25'].mean().unstack()
    fig_heat = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=['A: No Context', 'B: + Context', 'C: + Context + Glossary'],
        y=pivot.index.tolist(),
        colorscale=[[0,'#1e2130'],[0.5,'#3b5bdb'],[1,'#4ade80']],
        text=[[f'{v:.2f}' for v in row] for row in pivot.values],
        texttemplate='<b>%{text}</b>',
        textfont=dict(size=18, color='white'),
        showscale=True,
        colorbar=dict(tickfont=dict(color=FONT))
    ))
    fig_heat.update_layout(**base_layout('Score Heatmap: Model × Condition', height=280),
        xaxis=dict(tickfont=dict(size=11)),
        yaxis=dict(tickfont=dict(size=12))
    )
    st.plotly_chart(fig_heat, use_container_width=True)

# ===========================
# TAB 2: DIMENSIONS
# ===========================
with tab2:
    st.markdown('<div class="section-title">Per-Dimension Analysis</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        # Radar chart
        fig_radar = go.Figure()
        cats = ['Accuracy','Fluency','Cultural\nAppropr.','Terminology','Naturalness']
        for model in ['Claude','GPT4o']:
            row = model_dim[model_dim['Model']==model].iloc[0]
            vals = [row[d] for d in dims] + [row[dims[0]]]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals,
                theta=cats + [cats[0]],
                name=model,
                fill='toself',
                line_color=COLORS[model],
                fillcolor=COLORS[model],
                opacity=0.25
            ))
        fig_radar.update_layout(
            **base_layout('Dimension Scores by Model', height=380),
            polar=dict(
                radialaxis=dict(visible=True, range=[0,5], gridcolor=GRID,
                               tickfont=dict(size=9, color=FONT)),
                angularaxis=dict(tickfont=dict(size=10, color=FONT)),
                bgcolor=BG
            ),
            legend=dict(orientation='h', y=1.12, x=0.5, xanchor='center')
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_b:
        # Condition effect per dimension
        fig_dim = go.Figure()
        dim_short = ['Accuracy','Fluency','Cultural\nAppropr.','Terminology','Naturalness']
        for cond in ['A','B','C']:
            row = cond_dim[cond_dim['Condition']==cond].iloc[0]
            fig_dim.add_trace(go.Bar(
                name=COND_LABELS[cond],
                x=dim_short,
                y=[round(row[d],2) for d in dims],
                marker_color=COND_COLORS[cond],
                opacity=0.85
            ))
        fig_dim.update_layout(
            **base_layout('Dimension Scores by Condition', height=380),
            barmode='group',
            yaxis=dict(range=[2.5,5.5], title='Avg Score /5', gridcolor=GRID),
            xaxis=dict(gridcolor=GRID),
            legend=dict(orientation='h', y=1.18, x=0.5, xanchor='center')
        )
        st.plotly_chart(fig_dim, use_container_width=True)

    st.markdown("""<div class="insight-box">
    ⚡ <b>Key Finding:</b> Accuracy (avg 3.48/5) is the weakest dimension for both models —
    reflecting the difficulty of idioms, subject omission, and cultural nuance in KO→EN drama dialogue.
    Terminology (avg 4.52/5) is the strongest — both models handle character names and legal terms well.
    Context improves Accuracy most (+0.35 from A→C), while Fluency and Naturalness are less context-dependent.
    </div>""", unsafe_allow_html=True)

    # Dimension table
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Dimension Score Table</div>', unsafe_allow_html=True)
    dim_table = df.groupby(['Model','Condition'])[dims].mean().round(2)
    dim_table['Total'] = df.groupby(['Model','Condition'])['Total_25'].mean().round(2)
    st.dataframe(dim_table, use_container_width=True)

# ===========================
# TAB 3: MODEL COMPARISON
# ===========================
with tab3:
    st.markdown('<div class="section-title">Claude vs GPT-4o per String</div>', unsafe_allow_html=True)

    col_c, col_d = st.columns(2)

    with col_c:
        comp_reset = comp.reset_index()
        comp_reset['Winner'] = comp_reset.apply(
            lambda r: 'Claude' if r['Claude']>r['GPT4o']
            else ('GPT4o' if r['GPT4o']>r['Claude'] else 'Tie'), axis=1
        )
        fig_scatter = go.Figure()
        for winner, color in [('Claude','#a78bfa'),('GPT4o','#4ade80'),('Tie','#94A3B8')]:
            sub = comp_reset[comp_reset['Winner']==winner]
            count = len(sub)
            fig_scatter.add_trace(go.Scatter(
                x=sub['Claude'], y=sub['GPT4o'],
                mode='markers',
                name=f'{winner} ({count})',
                marker=dict(size=8, color=color, opacity=0.8),
                text=[f'ID {id_}' for id_ in sub['ID']],
                hovertemplate='%{text}<br>Claude: %{x:.1f}<br>GPT4o: %{y:.1f}<extra></extra>'
            ))
        fig_scatter.add_trace(go.Scatter(
            x=[5,25], y=[5,25], mode='lines',
            line=dict(dash='dot', color='#94A3B8', width=1),
            name='Equal', showlegend=True
        ))
        fig_scatter.update_layout(
            **base_layout('Claude vs GPT-4o per String (avg across conditions)', height=420),
            xaxis=dict(title='Claude Score', range=[5,26], gridcolor=GRID),
            yaxis=dict(title='GPT-4o Score', range=[5,26], gridcolor=GRID),
            legend=dict(orientation='h', y=-0.18, x=0.5, xanchor='center')
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_d:
        # Hardest strings
        hard = df.groupby(['ID','KO_Source'])['Total_25'].mean().reset_index()
        hard = hard.sort_values('Total_25').head(10)
        hard_colors = ['#f87171' if v < 10 else '#fbbf24' if v < 14 else '#4ade80'
                       for v in hard['Total_25']]
        fig_hard = go.Figure(go.Bar(
            x=hard['Total_25'].round(1),
            y=[f"ID {int(r['ID'])}: {str(r['KO_Source'])[:25]}..." for _,r in hard.iterrows()],
            orientation='h',
            marker_color=hard_colors,
            text=hard['Total_25'].round(1),
            textposition='outside',
            textfont=dict(size=11)
        ))
        fig_hard.update_layout(
            **base_layout('10 Most Challenging Strings'),
            xaxis=dict(range=[0,27], title='Avg Score /25', gridcolor=GRID),
            yaxis=dict(tickfont=dict(size=10))
        )
        st.plotly_chart(fig_hard, use_container_width=True)

    st.markdown("""<div class="insight-box">
    ⚡ <b>Key Finding:</b> Claude and GPT-4o are remarkably balanced — Claude wins on 21 strings,
    GPT-4o wins on 21, and 32 are ties. The hardest strings (ID 45, 18, 43) involve
    subject omission, figurative language, and sarcasm — exactly the cases where
    explicit context helps most.
    </div>""", unsafe_allow_html=True)

    # Filterable results table
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Browse Results by String</div>', unsafe_allow_html=True)

    col_filter1, col_filter2 = st.columns(2)
    with col_filter1:
        model_filter = st.selectbox("Filter by Model", ['All', 'Claude', 'GPT4o'])
    with col_filter2:
        cond_filter = st.selectbox("Filter by Condition", ['All', 'A', 'B', 'C'])

    filtered = df.copy()
    if model_filter != 'All':
        filtered = filtered[filtered['Model'] == model_filter]
    if cond_filter != 'All':
        filtered = filtered[filtered['Condition'] == cond_filter]

    display_cols = ['ID','KO_Source','Model','Condition'] + dims + ['Total_25','Notes']
    display_cols = [c for c in display_cols if c in filtered.columns]

    st.dataframe(
        filtered[display_cols].reset_index(drop=True),
        use_container_width=True,
        height=400,
        column_config={
            'Total_25': st.column_config.ProgressColumn('Total /25', min_value=0, max_value=25, format='%.0f'),
            'Accuracy': st.column_config.NumberColumn('Accuracy', format='%d'),
            'Fluency': st.column_config.NumberColumn('Fluency', format='%d'),
            'Cultural_Appropriateness': st.column_config.NumberColumn('Cultural', format='%d'),
            'Terminology': st.column_config.NumberColumn('Terminology', format='%d'),
            'Naturalness': st.column_config.NumberColumn('Naturalness', format='%d'),
        }
    )

# ===========================
# TAB 4: BLEU / TER
# ===========================
with tab4:
    st.markdown('<div class="section-title">Automated Metrics — BLEU & TER</div>', unsafe_allow_html=True)

    col_e, col_f = st.columns(2)

    with col_e:
        fig_bleu = go.Figure()
        for model in ['Claude','GPT4o']:
            sub = bleu_df[bleu_df['Model']==model]
            fig_bleu.add_trace(go.Bar(
                name=model,
                x=[COND_LABELS[c] for c in sub['Condition']],
                y=sub['BLEU'].round(2),
                marker_color=COLORS[model],
                opacity=0.85,
                text=sub['BLEU'].round(2),
                textposition='outside',
                textfont=dict(size=12)
            ))
        fig_bleu.update_layout(
            **base_layout('BLEU Score by Model and Condition (higher = better)', height=420),
            barmode='group',
            yaxis=dict(range=[8,14], title='BLEU Score', gridcolor=GRID),
            xaxis=dict(gridcolor=GRID),
            legend=dict(orientation='h', y=-0.18, x=0.5, xanchor='center')
        )
        st.plotly_chart(fig_bleu, use_container_width=True)

    with col_f:
        fig_ter = go.Figure()
        for model in ['Claude','GPT4o']:
            sub = bleu_df[bleu_df['Model']==model]
            fig_ter.add_trace(go.Bar(
                name=model,
                x=[COND_LABELS[c] for c in sub['Condition']],
                y=sub['TER'].round(2),
                marker_color=COLORS[model],
                opacity=0.85,
                text=sub['TER'].round(2),
                textposition='outside',
                textfont=dict(size=12)
            ))
        fig_ter.update_layout(
            **base_layout('TER Score by Model and Condition (lower = better)', height=420),
            barmode='group',
            yaxis=dict(range=[80,95], title='TER Score', gridcolor=GRID),
            xaxis=dict(gridcolor=GRID),
            legend=dict(orientation='h', y=-0.18, x=0.5, xanchor='center')
        )
        st.plotly_chart(fig_ter, use_container_width=True)

    # Human vs Automated comparison table
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Human Scores vs Automated Metrics</div>',
                unsafe_allow_html=True)

    human = df.groupby(['Model','Condition'])['Total_25'].mean().reset_index()
    human.columns = ['Model','Condition','Human_Score_25']
    comparison = pd.merge(human, bleu_df, on=['Model','Condition'])
    comparison['Human_Score_25'] = comparison['Human_Score_25'].round(2)
    comparison['Human_pct'] = (comparison['Human_Score_25']/25*100).round(1)

    st.dataframe(comparison, use_container_width=True, hide_index=True,
        column_config={
            'Human_Score_25': st.column_config.ProgressColumn('Human /25', min_value=0, max_value=25, format='%.2f'),
            'Human_pct': st.column_config.ProgressColumn('Human %', min_value=0, max_value=100, format='%.1f%%'),
            'BLEU': st.column_config.NumberColumn('BLEU ↑', format='%.2f'),
            'TER': st.column_config.NumberColumn('TER ↓', format='%.2f'),
        }
    )

    st.markdown("""<div class="warning-box">
    ⚠️ <b>Important Finding — Human vs Automated Metrics Disagree:</b><br><br>
    Human evaluation favors <b>Claude</b> (20.65 vs 20.41) while TER favors <b>GPT-4o</b> (lower TER = fewer edits needed).<br>
    Human evaluation shows <b>Condition C best</b> while BLEU shows <b>Condition C worst</b> for both models.<br><br>
    This is a well-documented phenomenon in NLP research — BLEU/TER measure surface-level word overlap
    with one reference translation, while human evaluation captures meaning, nuance, and cultural appropriateness.
    For creative dialogue translation with multiple valid outputs, automated metrics are unreliable as the sole
    quality indicator. This finding supports the case for <b>human-in-the-loop evaluation</b> in localization workflows.
    </div>""", unsafe_allow_html=True)

    st.markdown("""<div class="insight-box">
    📌 <b>Note on BLEU/TER for Dialogue:</b> Scores of 10-12 BLEU and 84-92 TER are expected for
    short drama dialogue lines — not a sign of poor translation quality. A line like
    <i>아니요 → "Nope"</i> scores 0 BLEU against reference <i>"No"</i> despite being a perfectly valid translation.
    Always interpret automated metrics comparatively (across conditions/models), never in isolation.
    </div>""", unsafe_allow_html=True)

# ===== FOOTER =====
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#4a5280;font-size:12px;font-family:'Space Mono',monospace;">
    LLM Translation Evaluation · Claude vs GPT-4o · KO→EN · Stranger S01E01 ·
    74 strings · 3 conditions · Human + Automated metrics
</div>
""", unsafe_allow_html=True)