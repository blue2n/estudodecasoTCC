import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification


def gerarDatasetDesbalanceado(saida, nome, n_samples, minoria_frac, random_state=None):
    """
    Gera dataset desbalanceado e salva CSV + gráfico.
    Retorna estatísticas.
    """
    n_minority = int(n_samples * minoria_frac)
    n_majority = n_samples - n_minority

    X, y = make_classification(
        n_classes=2,
        class_sep=2,
        weights=[n_majority / n_samples, n_minority / n_samples],
        n_informative=10,
        n_redundant=5,
        n_features=20,
        n_clusters_per_class=2,
        n_samples=n_samples,
        random_state=random_state,
    )

    contagem = np.bincount(y)
    total = contagem.sum()
    perc_major = contagem[0] / total * 100
    perc_minor = contagem[1] / total * 100

    print(f"{nome} -> Classe 0: {contagem[0]} ({perc_major:.2f}%) | Classe 1: {contagem[1]} ({perc_minor:.2f}%)")

    # Salvar dataset desbalanceado
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df["label"] = y
    os.makedirs(saida, exist_ok=True)
    path_csv = os.path.join(saida, f"{nome}Desbalanceado.csv")
    df.to_csv(path_csv, index=False)

    # Gráfico de distribuição
    plt.figure(figsize=(5, 4))
    plt.bar(["Classe 0 (majoritária)", "Classe 1 (minoritária)"], contagem, color=["royalblue", "lightgreen"])
    plt.title(f"Distribuição Desbalanceada - {nome}")
    plt.ylabel("Número de amostras")
    for i, val in enumerate(contagem):
        plt.text(i, val + (val * 0.01), f"{val}\n({val / total:.1%})", ha="center")
    plt.tight_layout()
    grafico_path = os.path.join(saida, f"{nome}Desbalanceado.png")
    plt.savefig(grafico_path)
    plt.close()

    return {
        "Nome": nome,
        "Amostras Totais": n_samples,
        "Classe 0": contagem[0],
        "Classe 1": contagem[1],
        "Classe 0 (%)": round(perc_major, 2),
        "Classe 1 (%)": round(perc_minor, 2),
        "Diferença Percentual": round(abs(perc_major - perc_minor), 2)
    }


def main(args):
    configs = [
        {"nome": "Dataset 100k", "n_samples": 100_000, "minority_frac": 0.10},  # 90% vs 10%
        {"nome": "Dataset 200k", "n_samples": 200_000, "minority_frac": 0.08},  # 92% vs 8%
        {"nome": "Dataset 300k", "n_samples": 300_000, "minority_frac": 0.06},  # 94% vs 6%
        {"nome": "Dataset 400k", "n_samples": 400_000, "minority_frac": 0.05},  # 95% vs 5%
        {"nome": "Dataset 500k", "n_samples": 500_000, "minority_frac": 0.03},  # 97% vs 3%
    ]

    resumo = []
    for cfg in configs:
        stats = gerarDatasetDesbalanceado(
            saida=args.saida,
            nome=cfg["nome"],
            n_samples=cfg["n_samples"],
            minoria_frac=cfg["minority_frac"],
            random_state=args.seed,
        )
        resumo.append(stats)

    # Salvar resumo
    resumo_df = pd.DataFrame(resumo)
    resumo_path = os.path.join(args.saida, "resumoDesbalanceado.csv")
    resumo_df.to_csv(resumo_path, index=False)
    print(f"\nResumo salvo em: {resumo_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--saida", type=str, required=True, help="Pasta de saída dos datasets desbalanceados")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
