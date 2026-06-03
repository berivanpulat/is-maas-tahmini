from pathlib import Path
import os
import sys

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor

matplotlib.use("Agg")

DATASET_CANDIDATES = [
    "job_salary_prediction_dataset.csv",
    "job-salary-prediction-dataset.csv",
]
REQUIRED_COLUMNS = {
    "experience_years",
    "education_level",
    "skills_count",
    "industry",
    "company_size",
    "remote_work",
    "salary",
}
NUMERIC_COLUMNS = {"experience_years", "skills_count", "salary", "certifications"}
TRAIN_SAMPLE_SIZE = 12000
TEST_SAMPLE_SIZE = 12000
CV_SAMPLE_SIZE = 3000
GRID_SAMPLE_SIZE = 3000
NB_SAMPLE_SIZE = 12000
SLOW_MODELS = {"KNN", "SVM", "AdaBoost", "Bagging", "Random Forest"}
OUTPUT_DIR = Path(__file__).resolve().parent / "ciktilar"


def locate_dataset(dataset_arg: str | None = None) -> Path:
    base_dir = Path(__file__).resolve().parent

    for candidate_value in [dataset_arg, os.environ.get("DATASET_PATH")]:
        if not candidate_value:
            continue

        candidate = Path(candidate_value).expanduser()
        if not candidate.is_absolute():
            candidate = (base_dir / candidate).resolve()

        if candidate.exists() and candidate.is_file():
            return candidate

    for filename in DATASET_CANDIDATES:
        direct_path = base_dir / filename
        if direct_path.exists():
            return direct_path

    for filename in DATASET_CANDIDATES:
        matches = sorted(base_dir.rglob(filename))
        if matches:
            return matches[0]

    raise FileNotFoundError("Uygun CSV veri dosyasi bulunamadi.")


def validate_columns(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Eksik sutunlar var: {', '.join(sorted(missing))}")


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cleaned_df = df.copy()

    for column in NUMERIC_COLUMNS.intersection(cleaned_df.columns):
        cleaned_df[column] = pd.to_numeric(cleaned_df[column], errors="coerce")

    cleaned_df = cleaned_df.dropna(subset=["salary"]).copy()
    if cleaned_df.empty:
        raise ValueError("Salary sutununda gecerli veri kalmadi.")

    for column in NUMERIC_COLUMNS.intersection(cleaned_df.columns) - {"salary"}:
        cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].median())

    text_columns = cleaned_df.select_dtypes(include=["object", "category", "string"]).columns
    for column in text_columns:
        cleaned_df[column] = cleaned_df[column].fillna("Unknown").astype(str).str.strip()
        cleaned_df[column] = cleaned_df[column].replace("", "Unknown")

    cleaned_df = cleaned_df.dropna().copy()
    if cleaned_df.empty:
        raise ValueError("Temizleme sonrasinda veri kalmadi.")

    return cleaned_df


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    categorical_cols = df.select_dtypes(
        include=["object", "category", "string", "bool"]
    ).columns.tolist()

    if "salary" in categorical_cols:
        categorical_cols.remove("salary")

    encoded_df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return encoded_df.drop("salary", axis=1), encoded_df["salary"]


def build_models() -> dict[str, object]:
    return {
        "Linear Regression": Pipeline(
            [("scaler", StandardScaler()), ("model", LinearRegression())]
        ),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=60,
            random_state=42,
            bootstrap=True,
            oob_score=True,
            n_jobs=1,
        ),
        "Bagging": BaggingRegressor(
            estimator=DecisionTreeRegressor(random_state=42),
            n_estimators=60,
            random_state=42,
            bootstrap=True,
            oob_score=True,
            n_jobs=1,
        ),
        "KNN": Pipeline(
            [("scaler", StandardScaler()), ("model", KNeighborsRegressor(n_neighbors=7))]
        ),
        "SVM": Pipeline(
            [("scaler", StandardScaler()), ("model", SVR(kernel="rbf", C=5, epsilon=0.2))]
        ),
        "AdaBoost": AdaBoostRegressor(
            n_estimators=80,
            learning_rate=0.1,
            random_state=42,
        ),
    }


def build_regression_weights(y: pd.Series, bins: int = 5) -> pd.Series:
    bucket_ids = pd.qcut(y, q=min(bins, y.nunique()), labels=False, duplicates="drop")
    bucket_counts = bucket_ids.value_counts()
    weights = bucket_ids.map(lambda bucket: 1.0 / bucket_counts[bucket])
    return weights / weights.mean()


def build_classification_weights(y: pd.Series) -> pd.Series:
    class_counts = y.value_counts()
    weights = y.map(lambda value: 1.0 / class_counts[value])
    return weights / weights.mean()


def save_plot() -> None:
    plt.tight_layout()
    OUTPUT_DIR.mkdir(exist_ok=True)
    filename = plt.gca().get_title().replace(" ", "_").lower()[:60] + ".png"
    plt.savefig(OUTPUT_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close()


def plot_regression_results(results_df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 6))
    ordered = results_df.sort_values("R2", ascending=False)
    plt.bar(ordered["Model"], ordered["R2"], color="teal")
    plt.title("Regresyon Model R2 Karsilastirmasi")
    plt.ylabel("R2")
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", alpha=0.3)
    save_plot()

    plt.figure(figsize=(10, 6))
    plt.bar(ordered["Model"], ordered["RMSE"], color="salmon")
    plt.title("Regresyon Model RMSE Karsilastirmasi")
    plt.ylabel("RMSE")
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", alpha=0.3)
    save_plot()


def plot_cv_results(cv_results: pd.DataFrame) -> None:
    plt.figure(figsize=(9, 5))
    ordered = cv_results.sort_values("CV Mean R2", ascending=False)
    plt.bar(ordered["Model"], ordered["CV Mean R2"], color="cornflowerblue")
    plt.title("Cross Validation Ortalama R2")
    plt.ylabel("CV Mean R2")
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", alpha=0.3)
    save_plot()


def plot_naive_bayes_results(test_accuracy: float, cv_accuracy: float) -> None:
    plt.figure(figsize=(6, 5))
    labels = ["Test Accuracy", "CV Mean Accuracy"]
    values = [test_accuracy, cv_accuracy]
    plt.bar(labels, values, color=["mediumpurple", "orange"])
    plt.title("Naive Bayes Basari Sonuclari")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.grid(axis="y", alpha=0.3)
    save_plot()


def build_prediction_table(
    best_model_name: str,
    best_model: object,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    limit: int = 10,
) -> pd.DataFrame:
    prediction_sample = X_test.head(limit)
    actual_sample = y_test.loc[prediction_sample.index]
    predicted_sample = best_model.predict(prediction_sample)

    prediction_table = pd.DataFrame(
        {
            "Gercek Salary": actual_sample.values,
            "Tahmin Edilen Salary": np.round(predicted_sample, 2),
            "Fark": np.round(actual_sample.values - predicted_sample, 2),
        },
        index=prediction_sample.index,
    )
    return prediction_table.reset_index(names="Satir No")


def save_prediction_table(prediction_table: pd.DataFrame) -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / "maas_tahmin_ornekleri.csv"
    prediction_table.to_csv(output_path, index=False)
    return output_path


def save_table_csv(dataframe: pd.DataFrame, filename: str) -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / filename
    dataframe.to_csv(output_path, index=False)
    return output_path


def save_table_image(dataframe: pd.DataFrame, title: str, filename: str) -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, max(3, len(dataframe) * 0.55)))
    ax.axis("off")
    rounded_df = dataframe.copy()

    for column in rounded_df.select_dtypes(include=[np.number]).columns:
        rounded_df[column] = rounded_df[column].round(4)

    table = ax.table(
        cellText=rounded_df.values,
        colLabels=rounded_df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.55)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#c7d1db")
        cell.set_linewidth(0.7)
        if row == 0:
            cell.set_facecolor("#244b6b")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        elif row % 2 == 0:
            cell.set_facecolor("#eef3f7")
        else:
            cell.set_facecolor("#ffffff")

    fig.patch.set_facecolor("white")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=14, color="#16324a")
    output_path = OUTPUT_DIR / filename
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_summary_table_image(
    grid_model_name: str,
    grid_best_score: float,
    grid_best_params: dict,
    nb_accuracy: float,
    nb_cv_accuracy: float,
) -> Path:
    summary_df = pd.DataFrame(
        [
            ["Grid Search", grid_model_name, round(grid_best_score, 4), str(grid_best_params)],
            ["Naive Bayes", "GaussianNB", round(nb_accuracy, 4), f"CV Mean Accuracy={nb_cv_accuracy:.4f}"],
        ],
        columns=["Bolum", "Model", "Skor", "Detay"],
    )
    return save_table_image(summary_df, "Ozet Sonuc Tablosu", "ozet_sonuc_tablosu.png")


def build_confusion_matrix_table(matrix: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        matrix,
        index=["Gercek Negatif (0)", "Gercek Pozitif (1)"],
        columns=["Tahmin Negatif (0)", "Tahmin Pozitif (1)"],
    ).reset_index(names="Sinif")


def save_confusion_matrix_heatmap(matrix: np.ndarray, title: str, filename: str) -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    heatmap = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Tahmin Negatif", "Tahmin Pozitif"])
    ax.set_yticks([0, 1], labels=["Gercek Negatif", "Gercek Pozitif"])
    ax.set_title(title, fontsize=12, fontweight="bold", color="#16324a")
    plt.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="#0b2239", fontsize=11, fontweight="bold")

    output_path = OUTPUT_DIR / filename
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def sample_if_needed(
    model_name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    fit_X = X_train
    fit_y = y_train
    pred_X = X_test
    pred_y = y_test

    if model_name in SLOW_MODELS and len(X_train) > TRAIN_SAMPLE_SIZE:
        fit_X = X_train.sample(TRAIN_SAMPLE_SIZE, random_state=42)
        fit_y = y_train.loc[fit_X.index]

    if model_name in SLOW_MODELS and len(X_test) > TEST_SAMPLE_SIZE:
        pred_X = X_test.sample(TEST_SAMPLE_SIZE, random_state=42)
        pred_y = y_test.loc[pred_X.index]

    return fit_X, pred_X, fit_y, pred_y


def fit_with_sample_weight(model_name: str, model: object, X: pd.DataFrame, y: pd.Series) -> None:
    sample_weight = build_regression_weights(y)
    fit_kwargs = {}

    if model_name in {"Linear Regression", "SVM"}:
        fit_kwargs["model__sample_weight"] = sample_weight
    elif model_name != "KNN":
        fit_kwargs["sample_weight"] = sample_weight

    model.fit(X, y, **fit_kwargs)


def evaluate_models(
    models: dict[str, object],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> pd.DataFrame:
    rows = []

    for model_name, model in models.items():
        fit_X, pred_X, fit_y, pred_y = sample_if_needed(
            model_name, X_train, X_test, y_train, y_test
        )
        fit_with_sample_weight(model_name, model, fit_X, fit_y)
        predictions = model.predict(pred_X)

        rows.append(
            {
                "Model": model_name,
                "MAE": mean_absolute_error(pred_y, predictions),
                "MSE": mean_squared_error(pred_y, predictions),
                "RMSE": np.sqrt(mean_squared_error(pred_y, predictions)),
                "R2": r2_score(pred_y, predictions),
                "OOB": getattr(model, "oob_score_", np.nan),
            }
        )

    return pd.DataFrame(rows).sort_values("R2", ascending=False).reset_index(drop=True)


def run_cross_validation(X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    sample_size = min(CV_SAMPLE_SIZE, len(X_train))
    X_sample = X_train.sample(sample_size, random_state=42)
    y_sample = y_train.loc[X_sample.index]

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    models = {
        "Linear Regression": Pipeline(
            [("scaler", StandardScaler()), ("model", LinearRegression())]
        ),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=25, random_state=42, n_jobs=1),
        "KNN": Pipeline(
            [("scaler", StandardScaler()), ("model", KNeighborsRegressor(n_neighbors=7))]
        ),
        "SVM": Pipeline(
            [("scaler", StandardScaler()), ("model", SVR(kernel="rbf", C=5, epsilon=0.2))]
        ),
    }

    rows = []
    for model_name, model in models.items():
        scores = cross_val_score(model, X_sample, y_sample, cv=cv, scoring="r2", n_jobs=1)
        rows.append(
            {
                "Model": model_name,
                "CV Mean R2": scores.mean(),
                "CV Std": scores.std(),
            }
        )

    return pd.DataFrame(rows).sort_values("CV Mean R2", ascending=False).reset_index(drop=True)


def run_grid_search(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[str, float, dict]:
    sample_size = min(GRID_SAMPLE_SIZE, len(X_train))
    X_sample = X_train.sample(sample_size, random_state=42)
    y_sample = y_train.loc[X_sample.index]

    grid = GridSearchCV(
        estimator=Pipeline(
            [("scaler", StandardScaler()), ("model", KNeighborsRegressor())]
        ),
        param_grid={
            "model__n_neighbors": [3, 5, 7, 9],
            "model__weights": ["uniform", "distance"],
        },
        cv=3,
        scoring="r2",
        n_jobs=1,
    )
    grid.fit(X_sample, y_sample)
    return "KNN", grid.best_score_, grid.best_params_


def evaluate_naive_bayes(X: pd.DataFrame, y: pd.Series) -> tuple[float, float]:
    y_binary = (y >= y.median()).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.2, random_state=42
    )

    sample_size = min(NB_SAMPLE_SIZE, len(X_train))
    X_train = X_train.sample(sample_size, random_state=42)
    y_train = y_train.loc[X_train.index]

    model = Pipeline([("scaler", StandardScaler()), ("model", GaussianNB())])
    model.fit(X_train, y_train, model__sample_weight=build_classification_weights(y_train))
    predictions = model.predict(X_test)

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=1)

    return accuracy_score(y_test, predictions), cv_scores.mean()


def evaluate_classification_models(X: pd.DataFrame, y: pd.Series) -> dict[str, dict]:
    y_binary = (y >= y.median()).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.2, random_state=42
    )

    sample_size = min(NB_SAMPLE_SIZE, len(X_train))
    X_train = X_train.sample(sample_size, random_state=42)
    y_train = y_train.loc[X_train.index]

    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=60, random_state=42, n_jobs=1),
        "Naive Bayes": Pipeline([("scaler", StandardScaler()), ("model", GaussianNB())]),
    }

    results = {}
    for model_name, model in models.items():
        if model_name == "Naive Bayes":
            model.fit(X_train, y_train, model__sample_weight=build_classification_weights(y_train))
        else:
            model.fit(X_train, y_train, sample_weight=build_classification_weights(y_train))

        predictions = model.predict(X_test)
        matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
        accuracy = accuracy_score(y_test, predictions)
        results[model_name] = {
            "accuracy": accuracy,
            "matrix": matrix,
        }

    return results


def main() -> None:
    dataset_arg = sys.argv[1] if len(sys.argv) > 1 else None
    dataset_path = locate_dataset(dataset_arg)

    df = pd.read_csv(dataset_path)
    validate_columns(df)
    df = prepare_dataframe(df)
    X, y = build_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = build_models()
    regression_results = evaluate_models(models, X_train, X_test, y_train, y_test)
    cv_results = run_cross_validation(X_train, y_train)
    grid_model_name, grid_best_score, grid_best_params = run_grid_search(X_train, y_train)
    nb_accuracy, nb_cv_accuracy = evaluate_naive_bayes(X, y)
    classification_results = evaluate_classification_models(X, y)
    plot_regression_results(regression_results)
    plot_cv_results(cv_results)
    plot_naive_bayes_results(nb_accuracy, nb_cv_accuracy)
    best_model_name = regression_results.iloc[0]["Model"]
    best_model = models[best_model_name]
    prediction_table = build_prediction_table(best_model_name, best_model, X_test, y_test)
    save_prediction_table(prediction_table)
    save_table_csv(regression_results, "regresyon_karsilastirma_tablosu.csv")
    save_table_csv(cv_results, "cross_validation_karsilastirma_tablosu.csv")
    save_table_image(
        regression_results,
        "Regresyon Karsilastirma Tablosu",
        "regresyon_karsilastirma_tablosu.png",
    )
    save_table_image(
        cv_results,
        "Cross Validation Karsilastirma Tablosu",
        "cross_validation_karsilastirma_tablosu.png",
    )
    save_table_image(
        prediction_table,
        "Ornek Maas Tahmin Tablosu",
        "ornek_maas_tahmin_tablosu.png",
    )
    save_summary_table_image(
        grid_model_name,
        grid_best_score,
        grid_best_params,
        nb_accuracy,
        nb_cv_accuracy,
    )
    confusion_outputs = []
    for model_name, result in classification_results.items():
        base_name = model_name.lower().replace(" ", "_")
        matrix_table = build_confusion_matrix_table(result["matrix"])
        save_table_csv(matrix_table, f"{base_name}_confusion_matrix.csv")
        save_table_image(
            matrix_table,
            f"{model_name} Confusion Matrix Tablosu",
            f"{base_name}_confusion_matrix_table.png",
        )
        save_confusion_matrix_heatmap(
            result["matrix"],
            f"{model_name} Confusion Matrix",
            f"{base_name}_confusion_matrix_heatmap.png",
        )
        confusion_outputs.append((model_name, result["accuracy"]))

   
    print("1. Regresyon Test Sonuclari")
    print(regression_results.to_string(index=False))

    print("\n2. Cross Validation Sonuclari")
    print(cv_results.to_string(index=False))

    print("\n3. Grid Search Sonucu")
    print(f"Model: {grid_model_name}")
    print(f"En iyi CV skoru: {grid_best_score:.4f}")
    print(f"En iyi parametreler: {grid_best_params}")

    print("\n4. Naive Bayes Sonucu")
    print(f"Test accuracy: {nb_accuracy:.4f}")
    print(f"CV mean accuracy: {nb_cv_accuracy:.4f}")

    print("\n5. Ornek Maas Tahminleri")
    print(f"En iyi model: {best_model_name}")
    print(prediction_table.to_string(index=False))

    print("\n6. Confusion Matrix Sonuclari")
    for model_name, accuracy in confusion_outputs:
        print(f"{model_name} accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as error:
        print(error)
        print("Kullanim: python proje.py <csv_dosyasi_yolu>")
        sys.exit(1)
    except ValueError as error:
        print(f"Veri hatasi: {error}")
        sys.exit(1)
