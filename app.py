import io
import time
import numpy as np
import pandas as pd
import streamlit as st
import arff
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder, label_binarize
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

st.set_page_config(
    page_title="MineScope Data Mining Lab",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.stApp {
    background:
      radial-gradient(circle at 5% 0%, rgba(0,210,255,.13), transparent 27%),
      radial-gradient(circle at 100% 0%, rgba(130,80,255,.13), transparent 27%),
      #050b14;
    color: #f4f8ff;
}
.block-container { max-width: 1450px; padding-top: 1.5rem; }
[data-testid="stHeader"] { background: transparent; }
.hero {
    padding: 32px;
    border-radius: 28px;
    border: 1px solid rgba(255,255,255,.09);
    background: linear-gradient(135deg, rgba(16,39,63,.96), rgba(7,15,27,.94));
    box-shadow: 0 24px 80px rgba(0,0,0,.25);
}
.badge {
    color: #66ddff;
    font-size: 12px;
    font-weight: 900;
    letter-spacing: .8px;
}
.hero h1 { font-size: 55px; margin: 10px 0; }
.gradient {
    background: linear-gradient(90deg,#fff,#61e0ff,#c6a7ff);
    -webkit-background-clip: text;
    color: transparent;
}
.muted { color:#94abc2; }
.card {
    padding: 18px;
    border-radius: 19px;
    min-height: 105px;
    border: 1px solid rgba(255,255,255,.08);
    background: rgba(12,28,46,.82);
}
.card-label { color:#8ea8c2; font-size:13px; }
.card-value { font-size:27px; font-weight:900; margin-top:8px; }
.output-box {
    padding: 18px;
    border-radius: 18px;
    background:#07101b;
    border:1px solid #1e3c58;
}
</style>
""", unsafe_allow_html=True)


def metric_card(label, value):
    st.markdown(
        f'<div class="card"><div class="card-label">{label}</div>'
        f'<div class="card-value">{value}</div></div>',
        unsafe_allow_html=True
    )


def decode(x):
    return x.decode("utf-8", errors="replace") if isinstance(x, bytes) else x


def load_dataset(uploaded):
    if uploaded.name.lower().endswith(".arff"):
        raw = uploaded.getvalue().decode("utf-8", errors="replace")
        obj = arff.load(io.StringIO(raw))
        attrs = obj["attributes"]
        cols = [a[0] for a in attrs]
        df = pd.DataFrame(obj["data"], columns=cols)
        for col in df.columns:
            df[col] = df[col].map(decode)
        return df, obj.get("relation", uploaded.name), attrs, "ARFF"

    df = pd.read_csv(uploaded)
    attrs = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            attrs.append((str(col), "numeric"))
        else:
            vals = sorted(df[col].dropna().astype(str).unique().tolist())
            attrs.append((str(col), vals))
    return df, uploaded.name.rsplit(".", 1)[0], attrs, "CSV"


def make_preprocessor(X):
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    parts = []

    if numeric:
        parts.append((
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            numeric
        ))

    if categorical:
        parts.append((
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]),
            categorical
        ))

    return ColumnTransformer(parts)


def make_model(name):
    if name == "ID3":
        return DecisionTreeClassifier(
            criterion="entropy",
            random_state=42
        )
    if name == "J48":
        return DecisionTreeClassifier(
            criterion="entropy",
            min_samples_leaf=1,
            random_state=42
        )
    if name == "Naive Bayes":
        return GaussianNB()
    return KNeighborsClassifier(n_neighbors=5)


def calculate_roc(y_true, probability, classes_count):
    if classes_count == 2:
        fpr, tpr, _ = roc_curve(y_true, probability[:, 1])
        return fpr, tpr, auc(fpr, tpr)

    y_bin = label_binarize(
        y_true,
        classes=np.arange(classes_count)
    )
    fpr, tpr, _ = roc_curve(
        y_bin.ravel(),
        probability.ravel()
    )
    return fpr, tpr, auc(fpr, tpr)


def train_and_evaluate(df, target, algorithm, test_option,
                       percentage, folds):
    X = df.drop(columns=[target])
    encoder = LabelEncoder()
    y = encoder.fit_transform(df[target].astype(str))

    if len(np.unique(y)) < 2:
        raise ValueError("The selected class attribute must contain at least two classes.")

    pipeline = Pipeline([
        ("preprocessor", make_preprocessor(X)),
        ("classifier", make_model(algorithm))
    ])

    start = time.perf_counter()

    if test_option == "Cross Validation":
        cv = StratifiedKFold(
            n_splits=folds,
            shuffle=True,
            random_state=42
        )

        predictions = cross_val_predict(
            pipeline, X, y, cv=cv, method="predict"
        )
        probabilities = cross_val_predict(
            pipeline, X, y, cv=cv, method="predict_proba"
        )

        pipeline.fit(X, y)

        y_test = y
        train_count = len(X)
        test_count = len(X)

        evaluation = f"{folds}-Fold Cross Validation"

    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=percentage / 100,
            random_state=42,
            stratify=y
        )

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)

        train_count = len(X_train)
        test_count = len(X_test)

        evaluation = (
            f"Percentage Split: "
            f"{100-percentage}% train / {percentage}% test"
        )

    elapsed_ms = (time.perf_counter() - start) * 1000

    n_classes = len(encoder.classes_)
    average = "binary" if n_classes == 2 else "weighted"

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions,
        average=average, zero_division=0
    )
    recall = recall_score(
        y_test, predictions,
        average=average, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions,
        average=average, zero_division=0
    )

    fpr, tpr, roc_auc = calculate_roc(
        y_test, probabilities, n_classes
    )

    cm = confusion_matrix(y_test, predictions)

    report = classification_report(
        y_test,
        predictions,
        target_names=[str(x) for x in encoder.classes_],
        zero_division=0,
        digits=4
    )

    tree = None
    feature_names = None

    if algorithm in ["ID3", "J48"]:
        tree = pipeline.named_steps["classifier"]
        try:
            feature_names = (
                pipeline.named_steps["preprocessor"]
                .get_feature_names_out()
            )
        except Exception:
            feature_names = None

    return {
        "algorithm": algorithm,
        "target": target,
        "evaluation": evaluation,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": roc_auc,
        "time": elapsed_ms,
        "fpr": fpr,
        "tpr": tpr,
        "cm": cm,
        "report": report,
        "classes": encoder.classes_,
        "tree": tree,
        "feature_names": feature_names,
        "train_count": train_count,
        "test_count": test_count
    }


# Header
st.markdown("""
<div class="hero">
    <div class="badge">⛏️ MINESCOPE · DATA MINING MINI PROJECT</div>
    <h1>Explore. <span class="gradient">Classify.</span> Evaluate.</h1>
    <p class="muted" style="font-size:18px">
        Upload your own ARFF or CSV dataset, inspect its Current Relation,
        select ID3, J48, Naive Bayes or KNN, choose the test method,
        and generate classification output, ROC curves and decision trees.
    </p>
</div>
""", unsafe_allow_html=True)


if "df" not in st.session_state:
    st.session_state.df = None
    st.session_state.results = {}
    st.session_state.relation = None
    st.session_state.attributes = None
    st.session_state.source = None


page = st.sidebar.radio(
    "Workspace",
    [
        "🧹 Preprocess",
        "🤖 Classify",
            "📈 ROC Curves",
        "🌳 Decision Tree"
    ]
)


# PREPROCESS
if page == "🧹 Preprocess":

    st.header("🧹 Preprocess")

    uploaded = st.file_uploader(
        "Upload Your Own Dataset",
        type=["arff", "csv"]
    )

    if uploaded:
        try:
            df, relation, attrs, source = load_dataset(uploaded)

            df.columns = [str(c).strip() for c in df.columns]

            st.session_state.df = df
            st.session_state.relation = relation
            st.session_state.attributes = attrs
            st.session_state.source = source
            st.session_state.results = {}

            st.success(f"{uploaded.name} uploaded successfully.")

        except Exception as e:
            st.error(f"Dataset loading error: {e}")

    if st.session_state.df is None:
        st.info(
            "Upload an ARFF or CSV dataset to see Current Relation information."
        )
        st.stop()

    df = st.session_state.df
    attrs = st.session_state.attributes

    st.subheader("📌 Current Relation")

    a, b, c, d = st.columns(4)

    with a:
        metric_card("Relation", st.session_state.relation)

    with b:
        metric_card("Attributes", len(attrs))

    with c:
        metric_card("Instances", len(df))

    with d:
        metric_card("Sum of Weights", f"{float(len(df)):.1f}")

    st.info(
        "For ordinary ARFF/CSV data, every instance is treated as weight 1.0; "
        "therefore Sum of Weights equals the number of instances."
    )

    st.subheader("Attribute Information")

    attribute_rows = []

    for name, attr_type in attrs:
        if isinstance(attr_type, (list, tuple)):
            display_type = "{" + ", ".join(
                str(x) for x in attr_type[:12]
            )
            if len(attr_type) > 12:
                display_type += ", ..."
            display_type += "}"
        else:
            display_type = str(attr_type)

        attribute_rows.append({
            "Attribute": str(name),
            "Type": display_type,
            "Missing": int(df[str(name)].isna().sum()),
            "Unique": int(df[str(name)].nunique(dropna=True))
        })

    st.dataframe(
        pd.DataFrame(attribute_rows),
        use_container_width=True
    )

    st.subheader("Dataset Preview")
    st.dataframe(df.head(25), use_container_width=True)

    x, y, z = st.columns(3)

    with x:
        metric_card(
            "Duplicate Instances",
            int(df.duplicated().sum())
        )

    with y:
        metric_card(
            "Missing Values",
            int(df.isna().sum().sum())
        )

    with z:
        metric_card(
            "Dataset Type",
            st.session_state.source
        )


# CLASSIFY
elif page == "🤖 Classify":

    st.header("🤖 Classify")

    if st.session_state.df is None:
        st.warning("First upload your dataset in Preprocess.")
        st.stop()

    df = st.session_state.df

    algorithm = st.selectbox(
        "Mining Algorithm",
        ["ID3", "J48", "Naive Bayes", "KNN"]
    )

    # Attribute selection removed.
    # The final column in the uploaded dataset is automatically used
    # as the class/target attribute.
    target = df.columns[-1]

    st.info(f"Class attribute is automatically selected as the last column: **{target}**")

    st.subheader("🧪 Test Option")

    test_option = st.radio(
        "Choose Evaluation Method",
        ["Cross Validation", "Percentage Split"],
        horizontal=True
    )

    if test_option == "Cross Validation":
        folds = 10
        percentage = 34
        st.info("Cross Validation is fixed at 10 folds.")
    else:
        folds = 10
        percentage = 34
        st.info("Percentage Split is fixed at 66% training / 34% testing.")

    # No Start Classification button: results are generated automatically.
    try:
        result = train_and_evaluate(
            df,
            target,
            algorithm,
            test_option,
            percentage,
            folds
        )
        st.session_state.results[algorithm] = result
        r = result
    except Exception as e:
        st.error(f"Classification failed: {e}")
        st.stop()

    st.success("Classification completed automatically.")

    st.subheader("📊 Performance Summary")

    cols = st.columns(6)
    values = [
        ("Accuracy", f"{r['accuracy']*100:.2f}%"),
        ("Precision", f"{r['precision']*100:.2f}%"),
        ("Recall", f"{r['recall']*100:.2f}%"),
        ("F1 Score", f"{r['f1']*100:.2f}%"),
        ("ROC AUC", f"{r['auc']:.4f}"),
        ("Decision Time", f"{r['time']:.2f} ms")
    ]

    for col, (label, value) in zip(cols, values):
        with col:
            metric_card(label, value)

    st.write(
        f"**Algorithm:** {r['algorithm']}  |  "
        f"**Class:** {r['target']}  |  "
        f"**Evaluation:** {r['evaluation']}"
    )

    st.write(
        f"**Training instances:** {r['train_count']}  |  "
        f"**Testing/evaluation instances:** {r['test_count']}"
    )

    st.subheader("📄 Classifier Output")

    classifier_output = f"""
=== CLASSIFIER OUTPUT ===

Algorithm              : {r['algorithm']}
Class Attribute        : {r['target']}
Test Option            : {r['evaluation']}
Training Instances     : {r['train_count']}
Testing Instances      : {r['test_count']}

=== PERFORMANCE ===

Accuracy               : {r['accuracy']:.4f}
Precision              : {r['precision']:.4f}
Recall                 : {r['recall']:.4f}
F1 Score               : {r['f1']:.4f}
ROC AUC                : {r['auc']:.4f}
Decision Time (ms)     : {r['time']:.4f}

=== CONFUSION MATRIX ===

{r['cm']}

=== CLASSIFICATION REPORT ===

{r['report']}
"""

    st.code(classifier_output, language="text")

    st.download_button(
        "⬇️ Download Classifier Output",
        data=classifier_output,
        file_name=f"{algorithm}_classifier_output.txt",
        mime="text/plain",
        key=f"download_{algorithm}"
    )

elif page == "📈 ROC Curves":

    st.header("📈 ROC Curves")

    if not st.session_state.results:
        st.info("Run classifiers first.")
        st.stop()

    selected = st.multiselect(
        "Algorithms to Compare",
        list(st.session_state.results.keys()),
        default=list(st.session_state.results.keys())
    )

    fig = go.Figure()

    comparison = []

    for algorithm in selected:

        r = st.session_state.results[algorithm]

        fig.add_trace(
            go.Scatter(
                x=r["fpr"],
                y=r["tpr"],
                mode="lines",
                name=f"{algorithm} (AUC={r['auc']:.3f})",
                line=dict(width=3)
            )
        )

        comparison.append({
            "Algorithm": algorithm,
            "Evaluation": r["evaluation"],
            "Accuracy": f"{r['accuracy']*100:.2f}%",
            "F1 Score": f"{r['f1']*100:.2f}%",
            "ROC AUC": f"{r['auc']:.4f}",
            "Decision Time (ms)": f"{r['time']:.2f}"
        })

    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random Baseline",
            line=dict(dash="dash")
        )
    )

    fig.update_layout(
        title="ROC Curve Comparison",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        template="plotly_dark",
        height=580
    )

    st.plotly_chart(fig, use_container_width=True)

    if comparison:
        st.subheader("Algorithm Comparison")
        st.dataframe(
            pd.DataFrame(comparison),
            use_container_width=True
        )


# TREE
else:

    st.header("🌳 Decision Tree")

    available = [
        x for x in ["ID3", "J48"]
        if x in st.session_state.results
    ]

    if not available:
        st.info(
            "Run ID3 or J48 from Classify first."
        )
        st.stop()

    selected = st.selectbox(
        "Choose Decision Tree",
        available
    )

    r = st.session_state.results[selected]
    tree = r["tree"]

    max_depth = max(1, tree.get_depth())

    display_depth = st.slider(
        "Display Depth",
        1,
        max_depth,
        min(4, max_depth)
    )

    fig, ax = plt.subplots(
        figsize=(22, 11)
    )

    plot_tree(
        tree,
        feature_names=r["feature_names"],
        class_names=[str(x) for x in r["classes"]],
        rounded=True,
        max_depth=display_depth,
        fontsize=8,
        ax=ax
    )

    ax.set_title(
        f"{selected} Decision Tree",
        fontsize=18
    )

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    a, b, c = st.columns(3)

    with a:
        metric_card("Tree Depth", tree.get_depth())

    with b:
        metric_card("Leaves", tree.get_n_leaves())

    with c:
        metric_card(
            "Decision Time",
            f"{r['time']:.2f} ms"
        )

st.divider()
st.caption(
    "MineScope Data Mining Lab · Python + Streamlit · "
    "ARFF/CSV · ID3 · J48 · Naive Bayes · KNN"
)
