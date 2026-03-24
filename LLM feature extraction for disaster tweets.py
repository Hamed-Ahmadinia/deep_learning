# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ### Hands-on 4
# ### Deep Learning
# ### Hamed Ahmadinia

# %%
from pathlib import Path
import re
import warnings

import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

warnings.filterwarnings("ignore")

# %% [markdown]
# ### 1. Configuration

# %%
CSV_PATH = "tweets.csv"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

# Start small; increase later if you want
N_TWEETS = 120
BATCH_SIZE = 8
MAX_NEW_TOKENS = 2

# 3-fold CV as requested in the exercise
N_SPLITS = 3
RANDOM_STATE = 42

# Output files
FEATURES_CSV = "hands_on_4_features.csv"
MISCLASSIFIED_CSV = "hands_on_4_misclassified.csv"
FIGURE_HIST = "hands_on_4_score_histograms.png"
FIGURE_MEAN = "hands_on_4_mean_scores.png"

# Candidate column names in tweets.csv
TEXT_COL_CANDIDATES = ["text", "tweet", "tweet_text", "message"]
LABEL_COL_CANDIDATES = ["target", "label", "class", "disaster"]


# %% [markdown]
# ### 2. Helper functions

# %%
def find_column(columns, candidates):
    """Find the first matching column name from a candidate list."""
    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def extract_score(text: str) -> int:
    """
    Extract the first digit 0-5 from the model output.
    If the model misbehaves, default to 0.
    """
    match = re.search(r"[0-5]", str(text))
    return int(match.group(0)) if match else 0


def run_question_batch(pipe, prompts, score_name):
    """
    Run one prompt set through the LLM and return numeric scores + raw outputs.
    """
    print(f"\nRunning prompt set: {score_name}")

    results = pipe(
        prompts,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        return_full_text=False,
        pad_token_id=pipe.tokenizer.eos_token_id,
    )

    scores = []
    raw_outputs = []

    for out in results:
        raw = out[0]["generated_text"].strip()
        raw_outputs.append(raw)
        scores.append(extract_score(raw))

    return scores, raw_outputs


def make_prompt_disaster(tweet_text: str) -> str:
    return (
        "You are a classifier.\n"
        "Task: Decide how likely this tweet is about a real disaster or emergency.\n"
        "Output exactly one character: 0 1 2 3 4 5.\n"
        "Do not explain.\n\n"
        f"Tweet: {tweet_text}\n\n"
        "Answer:"
    )


def make_prompt_damage(tweet_text: str) -> str:
    return (
        "You are a classifier.\n"
        "Task: Decide how much this tweet describes physical harm, damage, fire, violence, injury, or destruction.\n"
        "Output exactly one character: 0 1 2 3 4 5.\n"
        "Do not explain.\n\n"
        f"Tweet: {tweet_text}\n\n"
        "Answer:"
    )


def make_prompt_literal(tweet_text: str) -> str:
    return (
        "You are a classifier.\n"
        "Task: Decide whether this tweet is literal or metaphorical.\n"
        "0 = metaphorical or figurative.\n"
        "5 = literal real-world event.\n"
        "Output exactly one character: 0 1 2 3 4 5.\n"
        "Do not explain.\n\n"
        f"Tweet: {tweet_text}\n\n"
        "Answer:"
    )


def make_prompt_urgency(tweet_text: str) -> str:
    return (
        "You are a classifier.\n"
        "Task: Decide how urgent or emergency-like this tweet sounds.\n"
        "0 = not urgent.\n"
        "5 = highly urgent emergency situation.\n"
        "Output exactly one character: 0 1 2 3 4 5.\n"
        "Do not explain.\n\n"
        f"Tweet: {tweet_text}\n\n"
        "Answer:"
    )


def load_dataset(csv_path):
    """Load tweets.csv and normalize column names."""
    df = pd.read_csv(csv_path)

    print("Columns in dataset:", list(df.columns))

    text_col = find_column(df.columns, TEXT_COL_CANDIDATES)
    label_col = find_column(df.columns, LABEL_COL_CANDIDATES)

    if text_col is None:
        raise ValueError(
            f"Could not find tweet text column. Available columns: {list(df.columns)}"
        )

    if label_col is None:
        raise ValueError(
            f"Could not find label column. Available columns: {list(df.columns)}"
        )

    out = df[[text_col, label_col]].copy()
    out.columns = ["text", "label"]
    out = out.dropna(subset=["text", "label"]).reset_index(drop=True)

    # Normalize labels if they are strings
    if out["label"].dtype == object:
        label_map = {
            "0": 0,
            "1": 1,
            "false": 0,
            "true": 1,
            "no": 0,
            "yes": 1,
            "not disaster": 0,
            "disaster": 1,
            "non-disaster": 0,
        }
        out["label"] = (
            out["label"].astype(str).str.strip().str.lower().map(label_map)
        )

    out["label"] = out["label"].astype(int)
    return out


def make_figures(results_df):
    """Create and save simple figures."""
    # Histogram figure
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    score_columns = ["disaster_score", "damage_score", "literal_score", "urgency_score"]

    for ax, col in zip(axes, score_columns):
        ax.hist(
            results_df[col],
            bins=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
            edgecolor="black",
        )
        ax.set_title(col)
        ax.set_xlabel("Score")
        ax.set_ylabel("Count")
        ax.set_xticks([0, 1, 2, 3, 4, 5])

    plt.tight_layout()
    plt.savefig(FIGURE_HIST, dpi=150)
    plt.show()

    # Mean score per tweet
    plt.figure(figsize=(12, 5))
    plt.plot(range(1, len(results_df) + 1), results_df["mean_score"], marker="o")
    plt.xlabel("Tweet index")
    plt.ylabel("Mean score")
    plt.title("Mean LLM feature score per tweet")
    plt.xticks(range(1, len(results_df) + 1, max(1, len(results_df) // 12)))
    plt.tight_layout()
    plt.savefig(FIGURE_MEAN, dpi=150)
    plt.show()


def run_cv(results_df, feature_cols):
    """
    Run 3-fold cross-validation using LLM score features.
    Returns fold scores, predictions, and misclassified rows.
    """
    X = results_df[feature_cols].values
    y = results_df["label"].values

    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    fold_accuracies = []
    all_true = []
    all_pred = []
    misclassified_parts = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        fold_accuracies.append(acc)

        all_true.extend(y_test.tolist())
        all_pred.extend(y_pred.tolist())

        fold_df = results_df.iloc[test_idx].copy()
        fold_df["pred_label"] = y_pred
        fold_df["correct"] = (fold_df["label"] == fold_df["pred_label"]).astype(int)
        fold_df["fold"] = fold_idx

        misclassified_parts.append(fold_df[fold_df["correct"] == 0])

        print(f"Fold {fold_idx} accuracy: {acc:.4f}")

    misclassified_df = pd.concat(misclassified_parts, ignore_index=True)

    print("\n=== 3-fold CV summary ===")
    print("Fold accuracies:", [round(x, 4) for x in fold_accuracies])
    print(f"Mean accuracy: {sum(fold_accuracies) / len(fold_accuracies):.4f}")

    print("\nConfusion matrix:")
    print(confusion_matrix(all_true, all_pred))

    print("\nClassification report:")
    print(classification_report(all_true, all_pred, digits=4))

    return fold_accuracies, misclassified_df


# %% [markdown]
# ### 3. Load data

# %%
print("Loading dataset...")
df = load_dataset(CSV_PATH)

# Keep only first N rows for practical runtime
df = df.head(N_TWEETS).copy()
tweets = df["text"].tolist()

print(f"Loaded {len(df)} tweets")

# %% [markdown]
# ### 4. Load the LLM pipeline

# %%
print("\nLoading LLM...")
pipe = pipeline(
    "text-generation",
    model=MODEL_ID,
    dtype="auto",
    device_map="auto",
    batch_size=BATCH_SIZE,
)

# Required for decoder-only batch inference
pipe.tokenizer.padding_side = "left"
pipe.tokenizer.pad_token = pipe.tokenizer.eos_token

print("LLM ready.")

# %% [markdown]
# ### 5. Build prompt sets

# %%
questions_disaster = [make_prompt_disaster(t) for t in tweets]
questions_damage = [make_prompt_damage(t) for t in tweets]
questions_literal = [make_prompt_literal(t) for t in tweets]
questions_urgency = [make_prompt_urgency(t) for t in tweets]

# %% [markdown]
# ### 6. Run batched LLM scoring

# %%
disaster_scores, disaster_raw = run_question_batch(
    pipe, questions_disaster, "disaster_score"
)
damage_scores, damage_raw = run_question_batch(
    pipe, questions_damage, "damage_score"
)
literal_scores, literal_raw = run_question_batch(
    pipe, questions_literal, "literal_score"
)
urgency_scores, urgency_raw = run_question_batch(
    pipe, questions_urgency, "urgency_score"
)

# %% [markdown]
# ### 7. Build the feature table

# %%
results_df = pd.DataFrame({
    "text": tweets,
    "label": df["label"].values,
    "disaster_score": disaster_scores,
    "damage_score": damage_scores,
    "literal_score": literal_scores,
    "urgency_score": urgency_scores,
    "disaster_raw_output": disaster_raw,
    "damage_raw_output": damage_raw,
    "literal_raw_output": literal_raw,
    "urgency_raw_output": urgency_raw,
})

results_df["mean_score"] = (
    results_df["disaster_score"]
    + results_df["damage_score"]
    + results_df["literal_score"]
    + results_df["urgency_score"]
) / 4.0

print("\nPreview table:")
print(
    results_df[
        ["text", "label", "disaster_score", "damage_score", "literal_score", "urgency_score", "mean_score"]
    ].head(15).to_string(index=False)
)

# %% [markdown]
# ### 8. Figures

# %%
make_figures(results_df)

# %% [markdown]
# ### 9. 3-fold CV classification

# %%
feature_cols = ["disaster_score", "damage_score", "literal_score", "urgency_score"]
fold_accuracies, misclassified_df = run_cv(results_df, feature_cols)

# %% [markdown]
# ### 10. Save results

# %%
results_df.to_csv(FEATURES_CSV, index=False)
misclassified_df.to_csv(MISCLASSIFIED_CSV, index=False)

print("\nSaved files:")
print(f"- {FEATURES_CSV}")
print(f"- {MISCLASSIFIED_CSV}")
print(f"- {FIGURE_HIST}")
print(f"- {FIGURE_MEAN}")

# %% [markdown]
# ### 11. Show misclassified tweets

# %%
print("\nMisclassified tweets preview:")
if len(misclassified_df) == 0:
    print("No misclassified tweets in this sample.")
else:
    print(
        misclassified_df[
            ["fold", "text", "label", "pred_label", "disaster_score", "damage_score", "literal_score", "urgency_score"]
        ].head(20).to_string(index=False)
    )

# %%
