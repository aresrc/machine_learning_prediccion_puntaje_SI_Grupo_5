"""Entrenamiento y predicción del puntaje PAES de Matemática 1."""

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

ID = "MRUN"
TARGET = "MATE1_REG_ACTUAL"
SCORE_MIN, SCORE_MAX = 100, 1000
SEED = 42
ACADEMIC = {
    "PROMEDIO_NOTAS": (1, 7),
    "PTJE_NEM": (100, 1000),
    "PTJE_RANKING": (100, 1000),
    "ANYO_DE_EGRESO": (1900, 2025),
}
SCHOOL = {
    "RAMA_EDUCACIONAL": ("H1", "H2", "H3", "H4", "T1", "T2", "T3", "T4", "T5"),
    "DEPENDENCIA": tuple(map(str, range(1, 7))),
    "CODIGO_REGION_EGRESO": tuple(map(str, range(1, 17))),
}
SOCIAL = {
    "CODIGO_REGION_DOMICILIO": tuple(map(str, range(1, 17))),
    "INGRESO_PERCAPITA_GRUPO_FA": tuple(map(str, range(1, 11))),
}
CATEGORIES = {**SCHOOL, **SOCIAL}
FEATURES = [*ACADEMIC, *CATEGORIES]
METRICS = {
    "mae": ("neg_mean_absolute_error", mean_absolute_error, -1),
    "rmse": ("neg_root_mean_squared_error", root_mean_squared_error, -1),
    "r2": ("r2", r2_score, 1),
}


def clean_features(data):
    """Aplica reglas fijas de dominio, iguales para entrenamiento y predicción."""
    cleaned = pd.DataFrame(index=data.index)
    for name, (low, high) in ACADEMIC.items():
        values = pd.to_numeric(
            data[name].astype("string").str.strip().str.replace(",", ".", regex=False),
            errors="coerce",
        )
        cleaned[name] = values.where(values.between(low, high)).astype(float)
    for name, allowed in CATEGORIES.items():
        values = data[name].astype("string").str.strip().str.upper()
        cleaned[name] = values.where(values.isin(allowed), "SIN_DATO").astype(object)
    return cleaned


def read_source(path, columns):
    data = pd.read_csv(path, sep=";", usecols=[ID, *columns], dtype="string")
    data[ID] = data[ID].str.strip()
    if data[ID].isna().any() or data[ID].eq("").any() or data[ID].duplicated().any():
        raise ValueError(f"{path.name} debe tener un MRUN único y presente en cada fila.")
    return data


def load_data(scores_path, socioeconomic_path):
    scores = read_source(scores_path, [*ACADEMIC, *SCHOOL, TARGET])
    socioeconomic = read_source(socioeconomic_path, SOCIAL)
    target = pd.to_numeric(scores[TARGET].str.strip(), errors="coerce")
    valid = target.between(SCORE_MIN, SCORE_MAX).fillna(False)
    joined = scores.loc[valid].merge(
        socioeconomic, how="left", on=ID, validate="one_to_one", indicator=True
    )
    data_summary = {
        "registros_inscritos": len(scores),
        "registros_socioeconomicos": len(socioeconomic),
        "puntaje_vacio_o_no_numerico": int(target.isna().sum()),
        "puntaje_cero": int(target.eq(0).sum()),
        "puntaje_fuera_de_escala_distinto_de_cero": int(
            (target.notna() & target.ne(0) & ~valid).sum()
        ),
        "puntaje_valido": len(joined),
        "puntaje_valido_sin_cruce_socioeconomico": int(joined["_merge"].eq("left_only").sum()),
    }
    features = clean_features(joined)
    data_summary["faltantes_modelado"] = {
        name: int(features[name].isna().sum()) for name in ACADEMIC
    } | {
        name: int(features[name].eq("SIN_DATO").sum()) for name in CATEGORIES
    }
    return features, joined[TARGET].astype(float), data_summary


def build_models():
    preprocessing = ColumnTransformer(
        [
            ("notas", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), list(ACADEMIC)),
            ("categorias", OneHotEncoder(handle_unknown="ignore", sparse_output=False), list(CATEGORIES)),
        ]
    )
    return {
        "Mediana": DummyRegressor(strategy="median"),
        "Ridge": Pipeline([("preprocesamiento", preprocessing), ("regresor", Ridge(alpha=10.0))]),
        "HistGradientBoosting": Pipeline(
            [
                ("preprocesamiento", preprocessing),
                ("regresor", HistGradientBoostingRegressor(
                    max_iter=160, max_leaf_nodes=31, learning_rate=0.1,
                    early_stopping=False, random_state=SEED
                )),
            ]
        ),
    }


def plot_scores(metrics, output):
    names = metrics["modelo"].tolist()
    positions = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 4.4), layout="constrained")
    ax.errorbar(
        metrics["mae_cv_media"], positions + 0.12, xerr=metrics["mae_cv_de"], fmt="o", capsize=4,
        color="#12666a", label="Validación cruzada (media ± DE)"
    )
    ax.scatter(metrics["mae_prueba"], positions - 0.12, marker="D",
               color="#ae542a", label="Prueba 20 %")
    ax.set_yticks(positions, names)
    ax.invert_yaxis()
    ax.set_xlabel("Error absoluto medio (puntos PAES) · Menor es mejor")
    ax.set_title("Predicción del puntaje de Matemática 1")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="center right")
    fig.savefig(output, dpi=140)
    plt.close(fig)


def train(args):
    X, y, summary = load_data(args.puntajes, args.socioeconomico)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED
    )
    summary.update(semilla=SEED, entrenamiento=len(X_train), prueba=len(X_test), pliegues=3)
    cv = KFold(n_splits=summary["pliegues"], shuffle=True, random_state=SEED)
    scoring = {name: config[0] for name, config in METRICS.items()}
    models = build_models()
    rows, folds = [], []
    with threadpool_limits(limits=4):
        for name, model in models.items():
            print(f"Evaluando {name}…", flush=True)
            scores = cross_validate(model, X_train, y_train, scoring=scoring, cv=cv, error_score="raise")
            row = {"modelo": name}
            fold_rows = [{"modelo": name, "pliegue": i} for i in range(1, cv.n_splits + 1)]
            for metric, (_, _, sign) in METRICS.items():
                values = scores[f"test_{metric}"] * sign
                row[f"{metric}_cv_media"] = float(values.mean())
                row[f"{metric}_cv_de"] = float(values.std(ddof=1))
                for fold, value in zip(fold_rows, values, strict=True):
                    fold[metric] = float(value)
            rows.append(row)
            folds.extend(fold_rows)

        winner = min(rows, key=lambda row: row["mae_cv_media"])["modelo"]
        summary["modelo_seleccionado_por_mae_cv"] = winner
        for row in rows:
            model = models[row["modelo"]]
            model.fit(X_train, y_train)
            prediction = model.predict(X_test)
            for metric, (_, measure, _) in METRICS.items():
                row[f"{metric}_prueba"] = float(measure(y_test, prediction))

    output = args.salida
    output.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "metricas.csv", index=False, float_format="%.6f")
    pd.DataFrame(folds).to_csv(output / "pliegues.csv", index=False, float_format="%.6f")
    (output / "resumen.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    plot_scores(metrics, output / "comparacion_mae.png")
    joblib.dump({"nombre": winner, "modelo": models[winner]}, output / "modelo.joblib")
    print(metrics[["modelo", "mae_cv_media", "mae_prueba", "rmse_prueba", "r2_prueba"]].to_string(index=False))
    print(f"Modelo seleccionado: {winner}. Resultados en {output}.")


def predict(args):
    profile = json.loads(args.perfil.read_text(encoding="utf-8"))
    if not isinstance(profile, dict) or not profile or profile.keys() - set(FEATURES):
        raise ValueError("El perfil debe ser un objeto JSON con los campos de ejemplos/perfil.json.")
    cleaned = clean_features(pd.DataFrame([profile], columns=FEATURES))
    if cleaned[list(ACADEMIC)].isna().all(axis=None) and cleaned[list(CATEGORIES)].eq("SIN_DATO").all(axis=None):
        raise ValueError("El perfil debe contener por lo menos un valor válido.")
    bundle = joblib.load(args.modelo)
    prediction = float(bundle["modelo"].predict(cleaned)[0])
    print(json.dumps({"modelo": bundle["nombre"], "puntaje_estimado_m1": round(prediction, 2)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("entrenar", help="Validar y entrenar los modelos con los CSV de PAES.")
    training.add_argument("--puntajes", type=Path, required=True)
    training.add_argument("--socioeconomico", type=Path, required=True)
    training.add_argument("--salida", type=Path, default=Path("salida"))
    training.set_defaults(handler=train)
    prediction = commands.add_parser("predecir", help="Estimar M1 para un perfil JSON.")
    prediction.add_argument("--modelo", type=Path, default=Path("salida/modelo.joblib"))
    prediction.add_argument("--perfil", type=Path, required=True)
    prediction.set_defaults(handler=predict)
    args = parser.parse_args()
    try:
        args.handler(args)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
