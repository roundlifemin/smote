import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, NearMiss

from collections import Counter

# -------------------
# Sidebar - 사용자 선택
# -------------------
st.sidebar.title("🔧 설정")
selected_model = st.sidebar.selectbox("모델 선택", ["LogisticRegression", "RandomForest"])
selected_sampler = st.sidebar.selectbox("샘플링 기법 선택", ["SMOTE", "RandomOverSampler", "RandomUnderSampler", "TomekLinks", "NearMiss"])

# -------------------
# 데이터 생성
# -------------------
X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, 
                           n_redundant=0, n_clusters_per_class=1,
                           weights=[0.9, 0.1], flip_y=0, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

st.title("📊 불균형 데이터 분류 - Streamlit 앱")
st.write("Train 클래스 분포:", Counter(y_train))

# -------------------
# 샘플러 & 모델 정의
# -------------------
sampler_dict = {
    "SMOTE": SMOTE(random_state=42),
    "RandomOverSampler": RandomOverSampler(random_state=42),
    "RandomUnderSampler": RandomUnderSampler(random_state=42),
    "TomekLinks": TomekLinks(),
    "NearMiss": NearMiss()
}
model_dict = {
    "LogisticRegression": (LogisticRegression(solver='liblinear'), {'classifier__C': [0.1, 1, 10]}),
    "RandomForest": (RandomForestClassifier(random_state=42), {'classifier__n_estimators': [50, 100], 'classifier__max_depth': [3, 5, None]})
}

sampler = sampler_dict[selected_sampler]
model, param_grid = model_dict[selected_model]

# -------------------
# 모델 학습 및 평가
# -------------------
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('sampler', sampler),
    ('classifier', model)
])

grid = GridSearchCV(pipe, param_grid, scoring='f1', cv=5, n_jobs=-1)
grid.fit(X_train, y_train)
y_pred = grid.predict(X_test)

# -------------------
# 결과 출력
# -------------------
st.subheader("✅ Best F1 Score")
st.metric("F1 Score", f"{grid.best_score_:.4f}")
st.write("Best Parameters:", grid.best_params_)

st.subheader("📋 Classification Report")
st.text(classification_report(y_test, y_pred))

st.subheader("📌 Confusion Matrix")
fig, ax = plt.subplots()
disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=grid.classes_)
disp.plot(ax=ax, cmap='Blues')
st.pyplot(fig)
