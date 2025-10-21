import pandas as pd
import argparse
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report


def rodarMLDataset(dataset_csv):
    print(f"\nCarregando dataset: {dataset_csv}")
    df = pd.read_csv(dataset_csv)

    # Normaliza os nomes das colunas (sem espaços, tudo minúsculo)
    df.columns = [c.strip().lower() for c in df.columns]

    # Confere se tem a coluna 'label'
    if "label" not in df.columns:
        raise ValueError(
            f"Nenhuma coluna 'label' encontrada no dataset {dataset_csv}. "
            f"Colunas disponíveis: {list(df.columns)}"
        )

    X = df.drop("label", axis=1)
    y = df["label"]

    # divide em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    resultados = {}

    # Modelo 1: Random Forest
    print("Treinando RandomForest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    resultados["RandomForest"] = classification_report(y_test, rf.predict(X_test), output_dict=True)

    # Modelo 2: Logistic Regression
    print("Treinando LogisticRegression...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    resultados["LogisticRegression"] = classification_report(y_test, lr.predict(X_test), output_dict=True)

    # Modelo 3: MLP (Rede Neural)
    print("Treinando MLPClassifier...")
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
    mlp.fit(X_train, y_train)
    resultados["MLPClassifier"] = classification_report(y_test, mlp.predict(X_test), output_dict=True)

    # Modelo 4: KNN
    print("Treinando KNN...")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    resultados["KNN"] = classification_report(y_test, knn.predict(X_test), output_dict=True)

    # Modelo 5: XGBoost
    print("Treinando XGBoost...")
    xgb = XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        eval_metric="logloss"
    )
    xgb.fit(X_train, y_train)
    resultados["XGBoost"] = classification_report(y_test, xgb.predict(X_test), output_dict=True)

    # Modelo 6: Gradient Boosting
    print("Treinando GradientBoosting...")
    gb = GradientBoostingClassifier(random_state=42)
    gb.fit(X_train, y_train)
    resultados["GradientBoosting"] = classification_report(y_test, gb.predict(X_test), output_dict=True)

    return resultados


def main(args):
    todosResultados = []

    for arquivo in os.listdir(args.pasta):
        if arquivo.endswith(".csv") and "dataset" in arquivo.lower():
            caminho = os.path.join(args.pasta, arquivo)
            resultados = rodarMLDataset(caminho)

            for modelo, metricas in resultados.items():
                linha = {
                    "dataset": arquivo,
                    "modelo": modelo,
                    "accuracy": metricas["accuracy"],
                    "precision_0": metricas["0"]["precision"],
                    "recall_0": metricas["0"]["recall"],
                    "f1_0": metricas["0"]["f1-score"],
                    "precision_1": metricas["1"]["precision"],
                    "recall_1": metricas["1"]["recall"],
                    "f1_1": metricas["1"]["f1-score"],
                }
                todosResultados.append(linha)

    df_final = pd.DataFrame(todosResultados)
    df_final.to_csv(args.saida, index=False)
    print(f"\nResultados finais salvos em {args.saida}")

    # 🔹 Gerar gráfico comparativo
    print("Gerando gráfico comparativo...")
    plt.figure(figsize=(12, 6))

    # Média das métricas por modelo
    colunasNumericas = df_final.select_dtypes(include=["number"]).columns
    df_grouped = df_final.groupby("modelo")[colunasNumericas].mean()

    # Gráfico de barras
    df_grouped[["accuracy", "f1_0", "f1_1"]].plot(
        kind="bar", figsize=(12, 6), colormap="viridis", rot=45
    )

    plt.title("Comparação de Modelos - Accuracy e F1-score")
    plt.ylabel("Pontuação")
    plt.tight_layout()
    plt.savefig("comparacaoModelos.png")
    print("Gráfico salvo como comparacao_modelos.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pasta", type=str, required=True, help="Pasta com os datasets CSV")
    parser.add_argument("--saida", type=str, required=True, help="Arquivo CSV para salvar todos os resultados")
    args = parser.parse_args()
    main(args)
