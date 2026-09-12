import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (accuracy_score,recall_score,precision_score,f1_score,classification_report,confusion_matrix)

os.makedirs('model',exist_ok=True)
os.makedirs('results',exist_ok=True)

train_df = pd.read_csv('datasets/train.csv')
val_df = pd.read_csv('datasets/val.csv')
test_df = pd.read_csv('datasets/test.csv')

train_df = train_df.dropna(subset=['target'])
val_df = val_df.dropna(subset=['target'])
test_df = test_df.dropna(subset=['target'])

X_train = train_df.drop(columns=["target","row_idx"])
y_train = train_df["target"]

X_val = val_df.drop(columns=["target", "row_idx"])
y_val = val_df["target"]

X_test = test_df.drop(columns=["target", "row_idx"])
y_test = test_df["target"]


num_features = X_train.select_dtypes(include=['int64','float64']).columns
cat_features = X_train.select_dtypes(exclude=['int64','float64']).columns


num_transformer = Pipeline(
    steps=[
        ('imputer',SimpleImputer(strategy='median')),
        ('scalar',StandardScaler())
    ]
)

cat_transformer = Pipeline(
    steps=[
        ('imputer',SimpleImputer(strategy='most_frequent')),
        ('encoder',OneHotEncoder(handle_unknown='ignore'))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ('num',num_transformer,num_features),
        ('cat',cat_transformer,cat_features)
    ]
)

# 1.
logistic_model = Pipeline(
    steps=[
        ('preprocessor',preprocessor),
        ('classifier',LogisticRegression(
            max_iter=1000,class_weight='balanced',random_state=42
        ))
    ]
)

logistic_model.fit(X_train,y_train)
logistic_predictions = logistic_model.predict(X_val)

# 2.
decision_tree = Pipeline(
    steps=[
        ('preprocessor',preprocessor),
        ('classifier',DecisionTreeClassifier(
            max_depth=10,class_weight='balanced',random_state=42
        ))
    ]
)

decision_tree.fit(X_train,y_train)
tree_predictions = decision_tree.predict(X_val)

# 3.
rfc = Pipeline(
    steps=[
        ('preprocessor',preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=300,max_depth=15,class_weight='balanced',random_state=42
        ))
    ]
)

rfc.fit(X_train,y_train)
rfc_predictions = rfc.predict(X_val)

# 4.
xgb_model = Pipeline(
    steps=[
        ('preprocessor',preprocessor),
        ('classifier',XGBClassifier(
            n_estimators=300,max_depth=6,learning_rate=0.05,scale_pos_weight=10,random_state=42,eval_metric="logloss"
        ))
    ]
)

xgb_model.fit(X_train,y_train)
xgb_predictions = xgb_model.predict(X_val)


def evaluate_model(name,y_true,predictions):
    accuracy = accuracy_score(y_true,predictions)
    precision = precision_score(y_true,predictions)
    recall = recall_score(y_true,predictions)
    f1 = f1_score(y_true,predictions)
    
    return {
        'Model':name,
        "accuracy":accuracy,
        "precision":precision,
        "recall":recall,
        "f1":f1
    }
    
results = []

results.append(
    evaluate_model(
        "Logistic Regression",
        y_val,
        logistic_predictions
    )
)

results.append(
    evaluate_model(
        "Decision Tree",
        y_val,
        tree_predictions
    )
)

results.append(
    evaluate_model(
        "Random Forest",
        y_val,
        rfc_predictions
    )
)

results.append(
    evaluate_model(
        "XGBoost",
        y_val,
        xgb_predictions
    )
)

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))


results_plot = results_df.set_index("Model")

results_plot.plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title("Model Comparison")
plt.xlabel("Machine Learning Model")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.legend(title="Metrics")
plt.tight_layout()

plt.savefig(
    "results/model_comparison.png",
    dpi=300
)

plt.show()


C_values = [0.01, 0.1, 1, 10, 100]

logistic_results = []
for c in C_values:

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(
                C=c,
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            ))
        ]
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_val)

    accuracy = accuracy_score(y_val, predictions)
    precision = precision_score(y_val, predictions)
    recall = recall_score(y_val, predictions)
    f1 = f1_score(y_val, predictions)

    logistic_results.append({
        "C": c,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })


logistic_results_df = pd.DataFrame(logistic_results)
print(logistic_results_df.to_string(index=False))


best_row = logistic_results_df.loc[logistic_results_df["F1 Score"].idxmax()]

best_C = best_row["C"]

print("\nBest C value:", best_C)
print("Best F1 Score:", best_row["F1 Score"])


best_logistic_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            C=0.10,
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ))
    ]
)

final_train = pd.concat([train_df, val_df],axis=0)

X_final = final_train.drop(columns=["target", "row_idx"])
y_final = final_train["target"]

best_logistic_model.fit(X_final,y_final)
test_predictions = best_logistic_model.predict(X_test)

final_accuracy = accuracy_score(y_test,test_predictions)
final_precision = precision_score(y_test,test_predictions)
final_recall = recall_score(y_test,test_predictions)
final_f1 = f1_score(y_test,test_predictions)

print("Accuracy :", final_accuracy)
print("Precision:", final_precision)
print("Recall   :", final_recall)
print("F1 Score :", final_f1)

print(classification_report(y_test,test_predictions))

joblib.dump(best_logistic_model,"model/diabetes_logistic_model.pkl")
print("\nFinal model saved successfully!")