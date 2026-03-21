import io
import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="AI精准教学分析平台", page_icon="📚", layout="wide")

# =========================
# 基础配置
# =========================
REQUIRED_COLS = {
    "student_name": "学生姓名",
    "student_id": "学生ID",
    "question_id": "题目ID",
    "question_text": "题目内容",
    "subject": "学科",
    "knowledge_point": "知识点",
    "difficulty": "难度",
    "is_correct": "是否正确",
    "score": "本题得分",
    "full_score": "本题满分",
    "submit_time": "作答时间",
}

OPTIONAL_COLS = {
    "class_name": "班级",
    "exam_name": "考试/练习名称",
    "error_type": "错因类型(已有标注)",
    "ai_error_type": "AI错因类型",
    "ai_comment": "AI点评",
    "source": "来源APP",
}

ERROR_KEYWORDS = {
    "知识性错误": ["定义", "概念", "性质", "分类", "名称", "基础"],
    "计算性错误": ["计算", "公式", "换算", "数值", "求值", "单位"],
    "审题性错误": ["条件", "审题", "漏看", "范围", "不正确", "正确的是"],
    "理解性错误": ["原理", "推断", "解释", "原因", "应用", "迁移"],
    "实验性错误": ["实验", "装置", "步骤", "现象", "操作", "试剂"],
}

COLOR_SET = {
    "bg": "#081223",
    "card": "#0f1d35",
    "card_soft": "#122445",
    "border": "rgba(125, 170, 255, 0.16)",
    "text": "#eaf2ff",
    "muted": "#9db2d7",
    "accent": "#60a5fa",
    "accent2": "#34d399",
    "accent3": "#fbbf24",
    "danger": "#fb7185",
}


def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        .stApp {{
            background:
                radial-gradient(circle at top left, rgba(59,130,246,0.14), transparent 26%),
                radial-gradient(circle at top right, rgba(52,211,153,0.10), transparent 22%),
                linear-gradient(180deg, #081223 0%, #091427 100%);
            color: {COLOR_SET['text']};
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei', sans-serif;
        }}
        .block-container {{padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1500px;}}
        h1, h2, h3, h4 {{color: {COLOR_SET['text']}; letter-spacing: 0.2px;}}
        .hero {{
            background: linear-gradient(135deg, rgba(19,38,74,.96), rgba(18,31,59,.94));
            border: 1px solid {COLOR_SET['border']};
            border-radius: 24px;
            padding: 24px 28px;
            box-shadow: 0 20px 60px rgba(3, 8, 20, 0.35);
            margin-bottom: 18px;
        }}
        .hero-title {{font-size: 34px; font-weight: 800; margin-bottom: 8px;}}
        .hero-sub {{font-size: 15px; color: {COLOR_SET['muted']}; line-height: 1.7;}}
        .section-title {{font-size: 20px; font-weight: 700; margin: 6px 0 14px 0;}}
        .insight-box {{
            background: linear-gradient(180deg, rgba(17,29,52,.92), rgba(14,25,46,.92));
            border: 1px solid {COLOR_SET['border']};
            border-radius: 18px;
            padding: 16px 18px;
            min-height: 132px;
        }}
        .insight-title {{font-size: 15px; color: {COLOR_SET['muted']}; margin-bottom: 8px;}}
        .insight-value {{font-size: 26px; font-weight: 800; color: {COLOR_SET['text']}; margin-bottom: 6px;}}
        .insight-desc {{font-size: 13px; color: {COLOR_SET['muted']}; line-height: 1.65;}}
        .small-card {{
            background: rgba(15,29,53,0.88);
            border: 1px solid {COLOR_SET['border']};
            border-radius: 18px;
            padding: 14px 16px;
            height: 100%;
        }}
        .pill {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 12px;
            color: #d9e8ff;
            background: rgba(96,165,250,.16);
            border: 1px solid rgba(96,165,250,.18);
            margin-right: 8px;
        }}
        .stTabs [data-baseweb="tab-list"] {{gap: 10px;}}
        .stTabs [data-baseweb="tab"] {{
            background: rgba(15, 29, 53, 0.82);
            border: 1px solid {COLOR_SET['border']};
            border-radius: 14px;
            color: {COLOR_SET['text']};
            padding: 8px 16px;
        }}
        .stTabs [aria-selected="true"] {{
            background: linear-gradient(135deg, rgba(37,99,235,.45), rgba(16,185,129,.22));
        }}
        div[data-testid="stMetric"] {{
            background: rgba(15,29,53,0.88);
            border: 1px solid {COLOR_SET['border']};
            border-radius: 18px;
            padding: 10px 14px;
        }}
        div[data-testid="stMetric"] label {{color: {COLOR_SET['muted']};}}
        div[data-testid="stMetricValue"] {{color: {COLOR_SET['text']};}}
        div[data-testid="stDataFrame"] {{border-radius: 18px; overflow: hidden;}}
        .stAlert {{border-radius: 18px;}}
        </style>
        """,
        unsafe_allow_html=True,
    )


def chart_layout(title: str = "") -> Dict:
    return dict(
        title=dict(text=title, x=0.02, font=dict(size=16, color=COLOR_SET["text"])),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.02)",
        font=dict(color=COLOR_SET["text"], family="-apple-system, BlinkMacSystemFont, Segoe UI, PingFang SC, Microsoft YaHei, sans-serif"),
        margin=dict(l=16, r=16, t=52, b=16),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.15),
    )


# =========================
# 数据处理
# =========================
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {v: k for k, v in REQUIRED_COLS.items()}
    rename.update({v: k for k, v in OPTIONAL_COLS.items()})
    return df.rename(columns={c: rename.get(c, c) for c in df.columns})



def read_uploaded_file(file) -> pd.DataFrame:
    if file.name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file)

    for encoding in ["utf-8-sig", "utf-8", "gbk", "gb18030"]:
        try:
            file.seek(0)
            return pd.read_csv(file, encoding=encoding)
        except Exception:
            continue
    file.seek(0)
    return pd.read_csv(file)



def validate_df(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    missing = [col for col in REQUIRED_COLS if col not in df.columns]
    return len(missing) == 0, missing



def parse_binary(value):
    if pd.isna(value):
        return np.nan
    s = str(value).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "对", "正确", "是"}:
        return 1
    if s in {"0", "false", "f", "no", "n", "错", "错误", "否"}:
        return 0
    try:
        return int(float(s))
    except Exception:
        return np.nan



def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df.copy())

    for col in ["student_name", "student_id", "question_id", "question_text", "subject", "knowledge_point"]:
        df[col] = df[col].astype(str).replace("nan", "")

    for col in ["difficulty", "score", "full_score"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["is_correct"] = df["is_correct"].apply(parse_binary)
    df["submit_time"] = pd.to_datetime(df["submit_time"], errors="coerce")

    for col, default in {
        "class_name": "默认班级",
        "exam_name": "默认练习",
        "error_type": "",
        "ai_error_type": "",
        "ai_comment": "",
        "source": "未标注",
    }.items():
        if col not in df.columns:
            df[col] = default
        df[col] = df[col].fillna(default)

    df["wrong_flag"] = np.where(df["is_correct"] == 0, 1, 0)
    df["accuracy_item"] = np.where(df["full_score"] > 0, df["score"] / df["full_score"], np.nan)
    df["difficulty_group"] = pd.cut(
        df["difficulty"].fillna(3),
        bins=[-np.inf, 2, 3, np.inf],
        labels=["基础题", "中档题", "提升题"],
    )
    return df



def infer_error_type(row) -> str:
    text = f"{row.get('question_text','')} {row.get('knowledge_point','')} {row.get('ai_comment','')}"
    for label, kws in ERROR_KEYWORDS.items():
        if any(kw in text for kw in kws):
            return label
    if pd.notna(row.get("difficulty", np.nan)) and row.get("difficulty", 0) >= 4:
        return "理解性错误"
    return "知识性错误"



def apply_rule_ai(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mask = (df["wrong_flag"] == 1) & (df["ai_error_type"].astype(str).str.strip() == "")
    if mask.any():
        df.loc[mask, "ai_error_type"] = df.loc[mask].apply(infer_error_type, axis=1)
        df.loc[mask, "ai_comment"] = df.loc[mask].apply(
            lambda r: f"该题主要暴露出【{r['ai_error_type']}】问题，建议围绕知识点“{r['knowledge_point']}”做1次概念澄清 + 2道同类巩固题 + 1道变式迁移题。",
            axis=1,
        )
    return df



def load_sample_data() -> pd.DataFrame:
    rows = [
        ["九年级2班", "阶段测验1", "小王", "S01", "Q001", "将下列各组物质按酸、碱、盐顺序排列，正确的是", "化学", "物质分类", 2, 0, 0, 2, "2026-03-20 08:00:00", "定义记忆不牢"],
        ["九年级2班", "阶段测验1", "小王", "S01", "Q002", "下列叙述正确的是（摩尔质量相关）", "化学", "摩尔质量", 3, 0, 0, 2, "2026-03-20 08:02:00", "概念与单位混淆"],
        ["九年级2班", "阶段测验1", "小王玲", "S02", "Q001", "将下列各组物质按酸、碱、盐顺序排列，正确的是", "化学", "物质分类", 2, 1, 2, 2, "2026-03-20 08:03:00", ""],
        ["九年级2班", "阶段测验1", "小王玲", "S02", "Q003", "下列关于气体制备的说法正确的是", "化学", "气体制备", 4, 0, 0, 2, "2026-03-20 08:05:00", "实验装置识别不清"],
        ["九年级2班", "阶段测验1", "李明", "S03", "Q001", "将下列各组物质按酸、碱、盐顺序排列，正确的是", "化学", "物质分类", 2, 0, 0, 2, "2026-03-20 08:06:00", "分类依据混淆"],
        ["九年级2班", "阶段测验1", "李明", "S03", "Q004", "过滤操作中玻璃棒的作用是", "化学", "混合物分离", 3, 0, 0, 2, "2026-03-20 08:08:00", "实验步骤记忆不准"],
        ["九年级2班", "阶段测验1", "王晓敏", "S04", "Q002", "下列叙述正确的是（摩尔质量相关）", "化学", "摩尔质量", 3, 0, 0, 2, "2026-03-20 08:09:00", "公式与单位换算错误"],
        ["九年级2班", "阶段测验1", "王晓敏", "S04", "Q005", "下列关于二氧化碳性质的描述正确的是", "化学", "二氧化碳性质", 4, 0, 0, 2, "2026-03-20 08:10:00", "性质迁移应用不足"],
        ["九年级2班", "阶段测验1", "李对", "S05", "Q003", "下列关于气体制备的说法正确的是", "化学", "气体制备", 4, 0, 0, 2, "2026-03-20 08:12:00", "条件审读不完整"],
        ["九年级2班", "阶段测验1", "李对", "S05", "Q004", "过滤操作中玻璃棒的作用是", "化学", "混合物分离", 3, 1, 2, 2, "2026-03-20 08:13:00", ""],
        ["九年级2班", "阶段测验1", "小王", "S01", "Q005", "下列关于二氧化碳性质的描述正确的是", "化学", "二氧化碳性质", 4, 0, 0, 2, "2026-03-20 08:15:00", "原理理解不清"],
        ["九年级2班", "阶段测验1", "小王玲", "S02", "Q004", "过滤操作中玻璃棒的作用是", "化学", "混合物分离", 3, 0, 0, 2, "2026-03-20 08:16:00", "操作目的理解不足"],
    ]
    cols = ["班级", "考试/练习名称", "学生姓名", "学生ID", "题目ID", "题目内容", "学科", "知识点", "难度", "是否正确", "本题得分", "本题满分", "作答时间", "AI点评"]
    return pd.DataFrame(rows, columns=cols)


# =========================
# 分析函数
# =========================
def compute_overview(df: pd.DataFrame) -> Dict[str, float]:
    students = df["student_id"].nunique()
    questions = df["question_id"].nunique()
    attempts = len(df)
    wrongs = int(df["wrong_flag"].sum())
    accuracy = (1 - df["wrong_flag"].mean()) * 100 if attempts else 0

    stu_acc = df.groupby(["student_id", "student_name"])["wrong_flag"].mean().reset_index()
    stu_acc["acc"] = 1 - stu_acc["wrong_flag"]
    high_risk = int((stu_acc["acc"] < 0.6).sum()) if not stu_acc.empty else 0
    avg_wrong = wrongs / students if students else 0
    mastery_index = max(0, min(100, 0.65 * accuracy + 0.35 * (100 - min(avg_wrong * 15, 100))))

    return {
        "学生人数": students,
        "题目数量": questions,
        "总作答次数": attempts,
        "班级正确率": round(accuracy, 1),
        "人均错题": round(avg_wrong, 2),
        "需关注学生": high_risk,
        "班级掌握指数": round(mastery_index, 1),
    }



def get_wrong_df(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["wrong_flag"] == 1].copy()



def weak_knowledge(df: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
    wrong_df = get_wrong_df(df)
    total = df.groupby("knowledge_point").size().reset_index(name="attempts")
    wrong = wrong_df.groupby("knowledge_point").size().reset_index(name="wrongs")
    merged = total.merge(wrong, on="knowledge_point", how="left").fillna(0)
    merged["wrong_rate"] = merged["wrongs"] / merged["attempts"]
    return merged.sort_values(["wrongs", "wrong_rate"], ascending=False).head(top_n)



def error_type_distribution(df: pd.DataFrame) -> pd.DataFrame:
    wrong_df = get_wrong_df(df)
    if wrong_df.empty:
        return pd.DataFrame(columns=["ai_error_type", "cnt"])
    return wrong_df.groupby("ai_error_type").size().reset_index(name="cnt").sort_values("cnt", ascending=False)



def student_mastery(df: pd.DataFrame) -> pd.DataFrame:
    temp = (
        df.groupby(["student_id", "student_name"])
        .agg(
            attempts=("question_id", "size"),
            wrongs=("wrong_flag", "sum"),
            avg_score=("accuracy_item", "mean"),
        )
        .reset_index()
    )
    temp["accuracy"] = 1 - temp["wrongs"] / temp["attempts"]
    temp["level"] = pd.cut(
        temp["accuracy"],
        bins=[-np.inf, 0.6, 0.8, np.inf],
        labels=["重点帮扶", "需要巩固", "表现稳定"],
    )
    return temp.sort_values(["accuracy", "wrongs"], ascending=[True, False])



def question_diagnosis(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    q = (
        df.groupby(["question_id", "question_text", "knowledge_point"])
        .agg(attempts=("question_id", "size"), wrongs=("wrong_flag", "sum"), difficulty=("difficulty", "mean"))
        .reset_index()
    )
    q["wrong_rate"] = q["wrongs"] / q["attempts"]
    return q.sort_values(["wrongs", "wrong_rate", "difficulty"], ascending=False).head(top_n)



def difficulty_perf(df: pd.DataFrame) -> pd.DataFrame:
    res = (
        df.groupby("difficulty_group")
        .agg(attempts=("question_id", "size"), accuracy=("wrong_flag", lambda s: 1 - s.mean()))
        .reset_index()
    )
    res["accuracy"] = res["accuracy"] * 100
    return res



def student_knowledge_matrix(df: pd.DataFrame, top_students: int = 8, top_kps: int = 8) -> pd.DataFrame:
    wrong_df = get_wrong_df(df)
    if wrong_df.empty:
        return pd.DataFrame()
    s_order = wrong_df.groupby("student_name").size().sort_values(ascending=False).head(top_students).index.tolist()
    k_order = wrong_df.groupby("knowledge_point").size().sort_values(ascending=False).head(top_kps).index.tolist()
    pivot = (
        wrong_df[wrong_df["student_name"].isin(s_order) & wrong_df["knowledge_point"].isin(k_order)]
        .pivot_table(index="student_name", columns="knowledge_point", values="wrong_flag", aggfunc="sum", fill_value=0)
        .reindex(index=s_order, columns=k_order)
    )
    return pivot



def teaching_narrative(df: pd.DataFrame) -> str:
    wrong_df = get_wrong_df(df)
    if wrong_df.empty:
        return "当前筛选范围内未发现错题，说明该班级在本轮练习中的基础掌握较稳定。建议继续保留分层训练，并增加少量高阶变式题，用于检验迁移能力。"

    weak = weak_knowledge(df, top_n=3)
    students = student_mastery(df).head(3)
    diff = difficulty_perf(df)
    low_band = diff.sort_values("accuracy").iloc[0]

    text = (
        f"当前最需要优先处理的知识点是{ '、'.join(weak['knowledge_point'].tolist()) }，这些知识点的错题量和错题率同时偏高，属于讲评与二次训练的主战场。"
        f" 从题目层面看，{low_band['difficulty_group']}正确率最低，说明学生在该难度带的思维链条还不够稳。"
        f" 个体层面建议优先关注{ '、'.join(students['student_name'].tolist()) }等学生，先完成基础概念订正，再安排同类题巩固。"
        " 教学顺序建议按‘高频薄弱知识点 → 易混概念澄清 → 错因分组训练 → 小测回看’来组织。"
    )
    return text



def stratified_actions(df: pd.DataFrame) -> List[str]:
    weak = weak_knowledge(df, top_n=3)
    weak_names = "、".join(weak["knowledge_point"].tolist()) if not weak.empty else "当前主要薄弱知识点"
    return [
        f"围绕 {weak_names} 设计一页式讲评单：每个知识点只讲1个核心定义、1个易错点、1个典型题。",
        "讲评时不要按题号顺序逐题讲，而是按错因重新分组，先处理知识性错误和理解性错误，再处理计算与审题问题。",
        "对正确率低于60%的学生单独生成订正任务：每人3道基础巩固题 + 1道变式题，并在下一次课前抽查。",
        "针对提升题正确率偏低的情况，建议增加‘条件变化—方法不变’和‘条件不变—问法变化’两类迁移题。",
    ]



def llm_prompt_template(rows: pd.DataFrame) -> str:
    sample = rows[["student_name", "question_id", "question_text", "knowledge_point", "difficulty", "is_correct"]].head(30)
    data_json = sample.to_dict(orient="records")
    return f"""
你是一名中学精准教学分析助手。请基于学生作答记录输出严格 JSON：
{{
  "class_summary": "班级整体问题概述",
  "priority_knowledge_points": ["知识点1", "知识点2", "知识点3"],
  "student_groups": [{{"group":"重点帮扶/需要巩固/表现稳定","students":["张三","李四"],"advice":"建议"}}],
  "question_insights": [{{"question_id":"Q001","error_type":"知识性错误/计算性错误/审题性错误/理解性错误/实验性错误","comment":"点评"}}]
}}
数据如下：
{json.dumps(data_json, ensure_ascii=False, indent=2)}
只返回 JSON，不要返回 Markdown。
""".strip()



def make_excel_report(df: pd.DataFrame) -> io.BytesIO:
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="明细数据", index=False)
        weak_knowledge(df, top_n=20).to_excel(writer, sheet_name="知识点诊断", index=False)
        student_mastery(df).to_excel(writer, sheet_name="学生画像", index=False)
        question_diagnosis(df, top_n=30).to_excel(writer, sheet_name="题目诊断", index=False)
        error_type_distribution(df).to_excel(writer, sheet_name="错因分布", index=False)
    out.seek(0)
    return out


# =========================
# 图表
# =========================
def fig_error_donut(error_df: pd.DataFrame) -> go.Figure:
    fig = px.pie(error_df, names="ai_error_type", values="cnt", hole=0.68)
    fig.update_traces(textposition="inside", textinfo="percent", marker=dict(line=dict(width=2, color="#081223")))
    fig.update_layout(**chart_layout("错因结构分布"))
    return fig



def fig_weak_knowledge(weak_df: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        weak_df.sort_values("wrongs", ascending=True),
        x="wrongs",
        y="knowledge_point",
        orientation="h",
        text="wrongs",
        color="wrong_rate",
        color_continuous_scale=[[0, "#60a5fa"], [0.5, "#34d399"], [1, "#fb7185"]],
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(**chart_layout("薄弱知识点优先级"), xaxis_title="错题次数", yaxis_title="")
    fig.update_coloraxes(colorbar_title="错题率")
    return fig



def fig_difficulty_perf(diff_df: pd.DataFrame) -> go.Figure:
    fig = px.line(diff_df, x="difficulty_group", y="accuracy", markers=True)
    fig.update_traces(line=dict(width=3), marker=dict(size=10))
    fig.update_layout(**chart_layout("不同难度带正确率"), xaxis_title="难度层级", yaxis_title="正确率(%)", yaxis_range=[0, 100])
    return fig



def fig_student_heatmap(pivot: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale=[[0, "#0f274e"], [0.35, "#2563eb"], [0.7, "#34d399"], [1, "#fb7185"]],
            colorbar=dict(title="错题数"),
        )
    )
    fig.update_layout(**chart_layout("学生 × 知识点错题热力图"), xaxis_title="知识点", yaxis_title="学生")
    return fig



def fig_question_scatter(q_df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        q_df,
        x="attempts",
        y="wrong_rate",
        size="wrongs",
        color="difficulty",
        hover_data=["question_id", "knowledge_point"],
        text="question_id",
        color_continuous_scale=[[0, "#60a5fa"], [1, "#fbbf24"]],
    )
    fig.update_traces(textposition="top center")
    fig.update_layout(**chart_layout("题目诊断：参与度 × 错题率"), xaxis_title="作答人数", yaxis_title="错题率")
    fig.update_yaxes(tickformat=".0%")
    return fig


# =========================
# 页面主体
# =========================
inject_css()

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">📚 AI精准教学分析平台</div>
        <div class="hero-sub">
            将学生在做题 App 中产生的作答数据，自动转化为班级诊断、知识点预警、学生画像与讲评建议。<br>
            页面结构按照“先总览、再定位问题、最后落到教学动作”的逻辑设计，适合教师日常讲评、教研展示与学校汇报。
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## ⚙️ 数据与页面设置")
    mode = st.radio("数据来源", ["使用示例数据", "上传Excel/CSV"], index=0)
    show_prompt = st.toggle("显示大模型接入 Prompt", value=False)
    show_detail = st.toggle("显示原始明细表", value=False)
    st.caption("建议优先上传 .xlsx 文件，中文字段更稳定，不易乱码。")

    if mode == "上传Excel/CSV":
        file = st.file_uploader("上传做题导出文件", type=["xlsx", "xls", "csv"])
        if file is None:
            st.info("请先上传数据文件，或切换到示例数据查看效果。")
            st.stop()
        raw_df = read_uploaded_file(file)
    else:
        raw_df = load_sample_data()

raw_df = normalize_columns(raw_df)
valid, missing_cols = validate_df(raw_df)
if not valid:
    st.error("缺少必要字段：" + "、".join(REQUIRED_COLS[c] for c in missing_cols))
    st.dataframe(pd.DataFrame(columns=list(REQUIRED_COLS.values()) + list(OPTIONAL_COLS.values())), use_container_width=True)
    st.stop()

df = apply_rule_ai(clean_df(raw_df))

# 筛选区
filter_cols = st.columns([1, 1, 1, 1, 1.2])
with filter_cols[0]:
    class_opt = ["全部"] + sorted(df["class_name"].dropna().unique().tolist())
    selected_class = st.selectbox("班级", class_opt)
with filter_cols[1]:
    exam_opt = ["全部"] + sorted(df["exam_name"].dropna().unique().tolist())
    selected_exam = st.selectbox("考试/练习", exam_opt)
with filter_cols[2]:
    subject_opt = ["全部"] + sorted(df["subject"].dropna().unique().tolist())
    selected_subject = st.selectbox("学科", subject_opt)
with filter_cols[3]:
    source_opt = ["全部"] + sorted(df["source"].dropna().unique().tolist())
    selected_source = st.selectbox("来源", source_opt)
with filter_cols[4]:
    diff_options = sorted([x for x in df["difficulty"].dropna().unique().tolist()])
    selected_diff = st.multiselect("难度范围", diff_options, default=diff_options)

fdf = df.copy()
if selected_class != "全部":
    fdf = fdf[fdf["class_name"] == selected_class]
if selected_exam != "全部":
    fdf = fdf[fdf["exam_name"] == selected_exam]
if selected_subject != "全部":
    fdf = fdf[fdf["subject"] == selected_subject]
if selected_source != "全部":
    fdf = fdf[fdf["source"] == selected_source]
if selected_diff:
    fdf = fdf[fdf["difficulty"].isin(selected_diff)]

if fdf.empty:
    st.warning("当前筛选条件下没有数据，请调整筛选项。")
    st.stop()

overview = compute_overview(fdf)
wrong_df = get_wrong_df(fdf)
weak_df = weak_knowledge(fdf, top_n=8)
error_df = error_type_distribution(fdf)
student_df = student_mastery(fdf)
question_df = question_diagnosis(fdf, top_n=10)
diff_df = difficulty_perf(fdf)
pivot_df = student_knowledge_matrix(fdf, top_students=8, top_kps=8)

# KPI 总览
k1, k2, k3, k4 = st.columns(4)
k5, k6, k7, k8 = st.columns(4)
with k1:
    st.markdown(f"<div class='insight-box'><div class='insight-title'>学生人数</div><div class='insight-value'>{overview['学生人数']}</div><div class='insight-desc'>参与当前筛选范围内练习的学生总数。</div></div>", unsafe_allow_html=True)
with k2:
    st.markdown(f"<div class='insight-box'><div class='insight-title'>总作答次数</div><div class='insight-value'>{overview['总作答次数']}</div><div class='insight-desc'>用于判断本次分析的数据量是否充足。</div></div>", unsafe_allow_html=True)
with k3:
    st.markdown(f"<div class='insight-box'><div class='insight-title'>班级正确率</div><div class='insight-value'>{overview['班级正确率']}%</div><div class='insight-desc'>班级整体掌握水平的第一核心指标。</div></div>", unsafe_allow_html=True)
with k4:
    st.markdown(f"<div class='insight-box'><div class='insight-title'>班级掌握指数</div><div class='insight-value'>{overview['班级掌握指数']}</div><div class='insight-desc'>综合正确率与人均错题后形成的教学监测指标。</div></div>", unsafe_allow_html=True)
with k5:
    st.markdown(f"<div class='insight-box'><div class='insight-title'>题目数量</div><div class='insight-value'>{overview['题目数量']}</div><div class='insight-desc'>当前练习包含的独立题目数量。</div></div>", unsafe_allow_html=True)
with k6:
    st.markdown(f"<div class='insight-box'><div class='insight-title'>人均错题</div><div class='insight-value'>{overview['人均错题']}</div><div class='insight-desc'>适合用于横向比较不同班级与不同练习。</div></div>", unsafe_allow_html=True)
with k7:
    st.markdown(f"<div class='insight-box'><div class='insight-title'>需关注学生</div><div class='insight-value'>{overview['需关注学生']}</div><div class='insight-desc'>正确率低于60%的学生人数，可用于分层辅导。</div></div>", unsafe_allow_html=True)
with k8:
    wrong_ratio = round((wrong_df.shape[0] / fdf.shape[0]) * 100, 1) if len(fdf) else 0
    st.markdown(f"<div class='insight-box'><div class='insight-title'>错题占比</div><div class='insight-value'>{wrong_ratio}%</div><div class='insight-desc'>结合知识点和题目分析，判断讲评优先级。</div></div>", unsafe_allow_html=True)

# 快速结论
n1, n2 = st.columns([1.6, 1])
with n1:
    st.info(teaching_narrative(fdf))
with n2:
    st.markdown(
        f"""
        <div class="small-card">
            <div class="section-title">本页分析逻辑</div>
            <span class="pill">先看班级整体</span>
            <span class="pill">再看知识点</span>
            <span class="pill">再看学生分层</span>
            <span class="pill">最后落到讲评动作</span>
            <div style="margin-top:12px; color:{COLOR_SET['muted']}; line-height:1.75; font-size:13px;">
                页面不再使用单纯“炫酷图表堆叠”的方式，而是让每个图承担明确任务：<br>
                环形图看错因结构，条形图看薄弱知识点，折线图看难度带表现，热力图看学生与知识点交叉问题，散点图看题目讲评优先级。
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# 选项卡
tab1, tab2, tab3, tab4 = st.tabs(["班级总览", "知识点与题目诊断", "学生画像", "数据与导出"])

with tab1:
    c1, c2, c3 = st.columns([1.05, 1.15, 1.1])
    with c1:
        st.plotly_chart(fig_error_donut(error_df) if not error_df.empty else go.Figure().update_layout(**chart_layout("错因结构分布")), use_container_width=True)
    with c2:
        st.plotly_chart(fig_weak_knowledge(weak_df) if not weak_df.empty else go.Figure().update_layout(**chart_layout("薄弱知识点优先级")), use_container_width=True)
    with c3:
        st.plotly_chart(fig_difficulty_perf(diff_df), use_container_width=True)

    sub1, sub2 = st.columns([1.2, 0.9])
    with sub1:
        st.markdown("### 班级分层名单")
        display_student = student_df[["student_name", "attempts", "wrongs", "accuracy", "level"]].copy()
        display_student["accuracy"] = (display_student["accuracy"] * 100).round(1).astype(str) + "%"
        st.dataframe(display_student, use_container_width=True, hide_index=True)
    with sub2:
        st.markdown("### 教师行动建议")
        for idx, item in enumerate(stratified_actions(fdf), 1):
            st.markdown(f"**{idx}.** {item}")

with tab2:
    up1, up2 = st.columns([1.1, 1])
    with up1:
        if not pivot_df.empty:
            st.plotly_chart(fig_student_heatmap(pivot_df), use_container_width=True)
        else:
            st.plotly_chart(go.Figure().update_layout(**chart_layout("学生 × 知识点错题热力图")), use_container_width=True)
    with up2:
        st.plotly_chart(fig_question_scatter(question_df), use_container_width=True)

    st.markdown("### 题目讲评优先清单")
    q_display = question_df.copy()
    q_display["wrong_rate"] = (q_display["wrong_rate"] * 100).round(1).astype(str) + "%"
    q_display["difficulty"] = q_display["difficulty"].round(1)
    st.dataframe(q_display, use_container_width=True, hide_index=True)

with tab3:
    options = sorted(fdf["student_name"].dropna().unique().tolist())
    selected_student = st.selectbox("选择学生查看个体画像", options)
    s_df = fdf[fdf["student_name"] == selected_student].copy()
    s_wrong = get_wrong_df(s_df)

    a, b, c = st.columns(3)
    a.metric("作答次数", len(s_df))
    a.metric("错题数量", int(s_df["wrong_flag"].sum()))
    b.metric("个人正确率", f"{(1 - s_df['wrong_flag'].mean()) * 100:.1f}%" if len(s_df) else "0%")
    b.metric("平均得分率", f"{s_df['accuracy_item'].mean() * 100:.1f}%" if s_df['accuracy_item'].notna().any() else "0%")
    level = pd.cut(pd.Series([(1 - s_df['wrong_flag'].mean()) if len(s_df) else 0]), bins=[-np.inf, 0.6, 0.8, np.inf], labels=["重点帮扶", "需要巩固", "表现稳定"]).iloc[0]
    c.metric("当前分层", str(level))
    c.metric("主要薄弱知识点数", s_wrong["knowledge_point"].nunique())

    s1, s2 = st.columns([1, 1.15])
    with s1:
        if not s_wrong.empty:
            sk = s_wrong.groupby("knowledge_point").size().reset_index(name="cnt")
            fig = px.bar(sk.sort_values("cnt", ascending=True), x="cnt", y="knowledge_point", orientation="h", text="cnt")
            fig.update_layout(**chart_layout("该生薄弱知识点分布"), xaxis_title="错题数", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("该学生当前筛选范围内没有错题。")
    with s2:
        st.markdown("### 个体错题清单与AI建议")
        if not s_wrong.empty:
            st.dataframe(s_wrong[["question_id", "question_text", "knowledge_point", "ai_error_type", "ai_comment"]], use_container_width=True, hide_index=True)
        else:
            st.dataframe(pd.DataFrame(columns=["question_id", "question_text", "knowledge_point", "ai_error_type", "ai_comment"]), use_container_width=True, hide_index=True)

with tab4:
    report = make_excel_report(fdf)
    st.download_button(
        label="📥 下载分析结果 Excel",
        data=report,
        file_name="AI精准教学分析结果.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    if show_prompt:
        st.markdown("### 可接入大模型的 Prompt 模板")
        st.code(llm_prompt_template(wrong_df if not wrong_df.empty else fdf), language="text")
    if show_detail:
        st.markdown("### 原始明细数据")
        st.dataframe(fdf, use_container_width=True)
    st.markdown("### 字段要求")
    st.dataframe(pd.DataFrame(columns=list(REQUIRED_COLS.values()) + list(OPTIONAL_COLS.values())), use_container_width=True)
