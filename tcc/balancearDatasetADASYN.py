import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import ADASYN


def balancearDatasetADASYN(caminho_csv, saida, random_state=None):
    nome_base = os.path.basename(caminho_csv).replace("Desbalanceado.csv", "")
    df = pd.read_csv(caminho_csv)

    X = df.drop(columns=["label"])
    y = df["label"]

    contagem_antes = np.bincount(y)
    total_antes = len(y)

    ada = ADASYN(random_state=random_state)
    X_res, y_res = ada.fit_resample(X, y)

    contagem_depois = np.bincount(y_res)
    total_depois = len(y_res)

    os.makedirs(saida, exist_ok=True)
    path_csv = os.path.join(saida, f"{nome_base}Balanceado.csv")

    df_res = pd.DataFrame(X_res, columns=X.columns)
    df_res["label"] = y_res
    df_res.to_csv(path_csv, index=False)

    # Gráfico antes e depois
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].bar(["Classe 0", "Classe 1"], contagem_antes, color=["royalblue", "lightgreen"])
    ax[0].set_title("Antes do ADASYN")
    ax[1].bar(["Classe 0", "Classe 1"], contagem_depois, color=["royalblue", "lightgreen"])
    ax[1].set_title("Depois do ADASYN")
    for a in ax:
        a.set_ylabel("Amostras")
        for i, val in enumerate(a.patches):
            a.text(val.get_x() + 0.3, val.get_height() + (val.get_height() * 0.01), str(int(val.get_height())), ha="center")
    plt.suptitle(f"Balanceamento ADASYN - {nome_base}")
    plt.tight_layout()
    grafico_path = os.path.join(saida, f"{nome_base}Balanceado.png")
    plt.savefig(grafico_path)
    plt.close()

    return {
        "Nome": nome_base,
        "Antes Classe 0": contagem_antes[0],
        "Antes Classe 1": contagem_antes[1],
        "Depois Classe 0": contagem_depois[0],
        "Depois Classe 1": contagem_depois[1],
        "Diferença % Antes": round(abs((contagem_antes[0] - contagem_antes[1]) / total_antes * 100), 2),
        "Diferença % Depois": round(abs((contagem_depois[0] - contagem_depois[1]) / total_depois * 100), 2)
    }


def main(args):
    arquivos = [f for f in os.listdir(args.entrada) if f.endswith("Desbalanceado.csv")]
    resumo = []

    for arquivo in arquivos:
        caminho_csv = os.path.join(args.entrada, arquivo)
        print(f"Balanceando {arquivo} ...")
        stats = balancearDatasetADASYN(caminho_csv, args.saida, random_state=args.seed)
        resumo.append(stats)

    resumo_df = pd.DataFrame(resumo)
    resumo_path = os.path.join(args.saida, "resumoBalanceado.csv")
    resumo_df.to_csv(resumo_path, index=False)
    print(f"\n[OK] Resumo salvo em: {resumo_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--entrada", type=str, required=True, help="Pasta de entrada com datasets desbalanceados")
    parser.add_argument("--saida", type=str, required=True, help="Pasta de saída para salvar datasets balanceados")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
