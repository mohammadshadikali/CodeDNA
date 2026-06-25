"""Streamlit dashboard for CodeDNA.

Three pages (via sidebar):
1) Overview: total commits, total authors, total flagged commits (from features.csv and anomalies.csv)
2) Author Fingerprints: bar charts of average features per author
3) Flagged Commits: show `anomalies.csv` with flagged rows highlighted in red

The app loads `features.csv` and `anomalies.csv` from the project root.
"""
from pathlib import Path
from typing import List

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURES = ROOT / "features.csv"
DEFAULT_ANOMALIES = ROOT / "anomalies.csv"


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
	try:
		return pd.read_csv(path)
	except Exception:
		return pd.DataFrame()


def overview_page(features: pd.DataFrame, anomalies: pd.DataFrame):
	st.header("Overview")

	total_commits = len(features)
	total_authors = features['label'].nunique() if 'label' in features.columns else 0
	flagged = 0
	if not anomalies.empty and 'is_anomaly' in anomalies.columns:
		# coerce to boolean
		flagged = int(anomalies['is_anomaly'].astype(bool).sum())

	col1, col2, col3 = st.columns(3)
	col1.metric("Total commits", total_commits)
	col2.metric("Total authors", total_authors)
	col3.metric("Flagged commits", flagged)

	st.markdown("---")
	st.subheader("Sample commits")
	if not features.empty:
		st.dataframe(features.head(100))
	else:
		st.info("No features data available. Place features.csv in project root.")


def author_fingerprints_page(features: pd.DataFrame):
	st.header("Author Fingerprints")

	if features.empty:
		st.info("No features data available. Place features.csv in project root.")
		return

	# Select numeric feature columns
	ignore = {"commit_hash", "label"}
	numeric_cols: List[str] = [c for c in features.columns if c not in ignore]
	numeric_cols = [c for c in numeric_cols if pd.api.types.is_numeric_dtype(features[c])]

	if not numeric_cols:
		st.warning("No numeric features found to display.")
		return

	grouped = features.groupby('label')[numeric_cols].mean()

	st.subheader("Average feature values per author")
	st.write(grouped)

	st.subheader("Bar charts")
	selected_feature = st.selectbox("Feature to plot", numeric_cols, index=0)

	chart_data = grouped[[selected_feature]]
	st.bar_chart(chart_data)


def highlight_anomalies(row):
	try:
		is_anom = bool(row.get('is_anomaly', False))
	except Exception:
		is_anom = False
	if is_anom:
		return ['background-color: salmon'] * len(row)
	return [''] * len(row)


def flagged_commits_page(anomalies: pd.DataFrame):
	st.header("Flagged Commits")

	if anomalies.empty:
		st.info("No anomalies data available. Place anomalies.csv in project root.")
		return

	# Ensure boolean column
	if 'is_anomaly' in anomalies.columns:
		anomalies['is_anomaly'] = anomalies['is_anomaly'].apply(lambda x: str(x).lower() in ('true', '1', 'yes') or x is True)

	st.subheader("Anomalies table")
	# Use Styler to highlight anomalous rows
	try:
		styled = anomalies.style.apply(highlight_anomalies, axis=1)
		st.dataframe(styled, use_container_width=True)
	except Exception:
		# Fallback: show table and a count
		st.dataframe(anomalies)
		st.write(f"Flagged count: {int(anomalies['is_anomaly'].astype(bool).sum())}")


def main():
	st.title("CodeDNA Dashboard")

	features_path = st.sidebar.text_input("Features CSV path", str(DEFAULT_FEATURES))
	anomalies_path = st.sidebar.text_input("Anomalies CSV path", str(DEFAULT_ANOMALIES))

	features = load_csv(Path(features_path)) if features_path else pd.DataFrame()
	anomalies = load_csv(Path(anomalies_path)) if anomalies_path else pd.DataFrame()

	page = st.sidebar.radio("Page", ["Overview", "Author Fingerprints", "Flagged Commits"])

	if page == "Overview":
		overview_page(features, anomalies)
	elif page == "Author Fingerprints":
		author_fingerprints_page(features)
	else:
		flagged_commits_page(anomalies)


if __name__ == '__main__':
	main()

