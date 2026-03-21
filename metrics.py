import numpy as np
import pandas as pd


COLUMN_ALIASES = {
    "student_name": ["学生姓名", "学生", "姓名", "student_name", "student"],
    "student_id": ["学生id", "学生编号", "学号", "student_id"],
    "question_id": ["题目id", "试题id", "question_id", "item_id"],
    "question_text": ["题目内容", "题干", "question_text", "question"],
    "knowledge_point": ["知识点", "知识点名称", "考点", "knowledge_point"],
    "difficulty": ["难度", "题目难度", "difficulty"],
    "is_correct": ["是否正确", "正确与否", "答对与否", "is_correct", "correct"],
    "score": ["得分", "score"],
    "full_score": ["满分", "full_score", "总分值"],
    "submit_time": ["作答时间", "提交时间", "时间", "submit_time", "answer_time"],
}


def clean_columns(columns):
    return [str(c).strip().replace("\n", "") for c in columns]


def _resolve_columns(df: pd.DataFrame):
    mapping = {}
    lower_map = {str(col).strip().lower(): col for col in df.columns}
    for std_col, aliases in COLUMN_ALIASES.items():
        found = None
        for a in aliases:
            key = a.strip().lower()
            if key in lower_map:
                found = lower_map[key]
                break
        if found is not None:
            mapping[found] = std_col
    return mapping


def normalize_input_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mapping = _resolve_columns(df)
    df = df.rename(columns=mapping)

    required = ["student_name", "question_id", "knowledge_point", "is_correct"]
    missing = [x for x in required if x not in df.columns]
    if missing:
        raise ValueError(f"缺少必要字段：{', '.join(missing)}")

    if "difficulty" not in df.columns:
        df["difficulty"] = "中等"
    if "score" not in df.columns:
        df["score"] = np.where(_coerce_bool(df["is_correct"]), 1, 0)
    if "full_score" not in df.columns:
        df["full_score"] = 1
    if "submit_time" not in df.columns:
        df["submit_time"] = pd.RangeIndex(0, len(df))
    if "question_text" not in df.columns:
        df["question_text"] = df["question_id"].astype(str)

    df["student_name"] = df["student_name"].astype(str)
    df["question_id"] = df["question_id"].astype(str)
    df["knowledge_point"] = df["knowledge_point"].astype(str)
    df["difficulty"] = df["difficulty"].astype(str).replace({"nan": "中等"})
    df["is_correct"] = _coerce_bool(df["is_correct"])
    df["is_wrong"] = (~df["is_correct"]).astype(int)
    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0)
    df["full_score"] = pd.to_numeric(df["full_score"], errors="coerce").replace(0, 1).fillna(1)
    df["score_rate"] = (df["score"] / df["full_score"]).clip(0, 1)
    try:
        df["submit_time"] = pd.to_datetime(df["submit_time"], errors="coerce")
    except Exception:
        pass
    return df


def _coerce_bool(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    true_vals = {"1", "true", "t", "yes", "y", "对", "正确", "答对"}
    false_vals = {"0", "false", "f", "no", "n", "错", "错误", "答错"}
    out = []
    for x in s:
        if x in true_vals:
            out.append(True)
        elif x in false_vals:
            out.append(False)
        else:
            try:
                out.append(float(x) > 0)
            except Exception:
                out.append(False)
    return pd.Series(out, index=series.index)


def difficulty_weight(x: str) -> float:
    text = str(x)
    if any(k in text for k in ["难", "困难", "hard", "高"]):
        return 1.5
    if any(k in text for k in ["中", "medium"]):
        return 1.2
    return 1.0


def calc_class_metrics(df: pd.DataFrame) -> dict:
    acc = df["is_correct"].mean()
    score_rate = df["score_rate"].mean()
    wrong_count = int(df["is_wrong"].sum())
    student_count = int(df["student_name"].nunique())
    attempt_count = int(len(df))
    mastery_index = round((0.7 * acc + 0.3 * score_rate) * 100, 1)
    risk_student_count = int(calc_student_risk(df).query("risk_level == '高风险'")["student_name"].nunique())
    return {
        "accuracy_rate": acc,
        "mastery_index": mastery_index,
        "student_count": student_count,
        "attempt_count": attempt_count,
        "wrong_count": wrong_count,
        "risk_student_count": risk_student_count,
    }


def calc_knowledge_priority(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    tmp["weight"] = tmp["difficulty"].apply(difficulty_weight)
    result = (
        tmp.groupby("knowledge_point")
        .agg(
            total_attempts=("question_id", "count"),
            wrong_attempts=("is_wrong", "sum"),
            error_students=("student_name", lambda x: x[tmp.loc[x.index, "is_wrong"] == 1].nunique()),
            question_weight=("weight", "mean"),
        )
        .reset_index()
    )
    result["error_rate"] = result["wrong_attempts"] / result["total_attempts"].replace(0, 1)
    result["priority_index"] = (result["error_rate"] * result["error_students"] * result["question_weight"]).round(3)
    result = result.sort_values(["priority_index", "error_students", "error_rate"], ascending=False).reset_index(drop=True)
    return result


def _longest_error_streak(group: pd.DataFrame) -> int:
    g = group.sort_values("submit_time").copy()
    vals = g["is_wrong"].tolist()
    best = cur = 0
    for v in vals:
        if v == 1:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def calc_student_risk(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    tmp["weight"] = tmp["difficulty"].apply(difficulty_weight)
    tmp["hard_wrong_weight"] = tmp["is_wrong"] * tmp["weight"]

    streaks = tmp.groupby("student_name").apply(_longest_error_streak).reset_index(name="longest_wrong_streak")
    result = (
        tmp.groupby("student_name")
        .agg(
            wrong_count=("is_wrong", "sum"),
            hard_error_weight=("hard_wrong_weight", "sum"),
            accuracy_rate=("is_correct", "mean"),
        )
        .reset_index()
        .merge(streaks, on="student_name", how="left")
    )
    result["continuous_error_penalty"] = result["longest_wrong_streak"] * 1.5
    result["risk_index"] = (result["wrong_count"] + result["continuous_error_penalty"] + result["hard_error_weight"]).round(2)

    p50 = result["risk_index"].quantile(0.5)
    p80 = result["risk_index"].quantile(0.8)

    def label(x):
        if x >= p80:
            return "高风险"
        if x >= p50:
            return "波动层"
        return "稳定层"

    result["risk_level"] = result["risk_index"].apply(label)
    return result.sort_values("risk_index", ascending=False).reset_index(drop=True)


def calc_item_discrimination(df: pd.DataFrame) -> pd.DataFrame:
    stu = (
        df.groupby("student_name")
        .agg(total_score=("score", "sum"), total_full=("full_score", "sum"))
        .reset_index()
    )
    stu["score_rate"] = stu["total_score"] / stu["total_full"].replace(0, 1)
    stu = stu.sort_values("score_rate", ascending=False).reset_index(drop=True)
    n = max(1, int(len(stu) * 0.27))
    high_students = set(stu.head(n)["student_name"])
    low_students = set(stu.tail(n)["student_name"])

    tmp = df.copy()
    high = (
        tmp[tmp["student_name"].isin(high_students)]
        .groupby("question_id")["is_correct"]
        .mean()
        .reset_index(name="high_group_acc")
    )
    low = (
        tmp[tmp["student_name"].isin(low_students)]
        .groupby("question_id")["is_correct"]
        .mean()
        .reset_index(name="low_group_acc")
    )
    kp = tmp.groupby("question_id").agg(knowledge_point=("knowledge_point", "first")).reset_index()
    result = kp.merge(high, on="question_id", how="left").merge(low, on="question_id", how="left")
    result[["high_group_acc", "low_group_acc"]] = result[["high_group_acc", "low_group_acc"]].fillna(0)
    result["discrimination"] = (result["high_group_acc"] - result["low_group_acc"]).round(3)
    return result.sort_values("discrimination", ascending=False).reset_index(drop=True)


def calc_teaching_gain(priority_df: pd.DataFrame) -> pd.DataFrame:
    result = priority_df[["knowledge_point", "error_rate", "error_students"]].copy()
    result["wrong_students"] = result["error_students"]
    result["avg_improvement_space"] = (1 - result["error_rate"]).round(3)
    result["teaching_gain"] = (result["wrong_students"] * result["avg_improvement_space"]).round(3)
    result = result.sort_values("teaching_gain", ascending=False).reset_index(drop=True)
    return result


def calc_difficulty_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    result = (
        df.groupby("difficulty")
        .agg(accuracy_rate=("is_correct", "mean"), attempts=("question_id", "count"))
        .reset_index()
    )
    result = result.sort_values("accuracy_rate", ascending=True).reset_index(drop=True)
    return result


def build_action_suggestions(class_metrics, priority_df, risk_df, disc_df, gain_df):
    top_kps = priority_df.head(3)["knowledge_point"].tolist()
    high_risk_count = int((risk_df["risk_level"] == "高风险").sum())
    top_gain = gain_df.head(3)["knowledge_point"].tolist()
    top_items = disc_df.head(3)["question_id"].tolist()

    class_conclusion = (
        f"本班当前正确率为 {class_metrics['accuracy_rate']:.1%}，掌握指数为 {class_metrics['mastery_index']:.1f}。"
        f"整体上更适合采用“共性薄弱点优先讲评 + 风险学生追踪补救”的教学策略。"
    )
    priority_conclusion = (
        f"当前最优先处理的知识点为 {', '.join(top_kps) if top_kps else '暂无'}，"
        f"它们同时具备较高错误率、较广覆盖学生数和较高教学处理价值。"
    )
    risk_conclusion = (
        f"当前识别出高风险学生 {high_risk_count} 人。建议优先对风险指数最高的学生进行连续错误清理和难题补救。"
    )
    action_conclusion = (
        f"建议本轮优先处理 {', '.join(top_gain) if top_gain else '暂无'}，并配合高区分度题目 {', '.join(top_items) if top_items else '暂无'} 进行二次训练，"
        f"以提升讲评效率和课堂收益。"
    )

    class_actions = [
        f"优先讲评知识点：{', '.join(top_kps[:2]) if top_kps else '暂无'}。",
        "先处理高覆盖错误，再处理低频个性错误，避免课堂重心分散。",
        "课堂讲评后安排 5~10 分钟同类变式检测，验证是否真正过关。",
    ]
    layered_actions = [
        f"高风险层：优先关注前 {min(5, len(risk_df))} 名风险学生，进行错因面谈与过关。",
        "波动层：围绕中档题和易混知识点做小组巩固，减少反复失误。",
        "稳定层：补充分层提高题，防止课堂内容过度下沉。",
    ]
    item_actions = [
        f"优先复用高区分度题：{', '.join(top_items) if top_items else '暂无'}。",
        f"优先投入高收益知识点：{', '.join(top_gain[:2]) if top_gain else '暂无'}。",
        "将低区分度或异常题单独标记，避免在公开讲评中占用过多时间。",
    ]

    return {
        "class_conclusion": class_conclusion,
        "priority_conclusion": priority_conclusion,
        "risk_conclusion": risk_conclusion,
        "action_conclusion": action_conclusion,
        "class_actions": class_actions,
        "layered_actions": layered_actions,
        "item_actions": item_actions,
    }
