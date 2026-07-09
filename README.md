# recsys-ecommerce

Sistema de recomendação de produtos para e-commerce baseado no comportamento de navegação dos usuários — **FIAP Pós Tech MLE, Tech Challenge Fase 2**.

Rede neural (NCF — Neural Collaborative Filtering) treinada com PyTorch sobre o dataset RetailRocket, com pipeline reprodutível: dados versionados com **DVC** (remote em Azure Blob), experimentos rastreados no **MLflow** (com Model Registry), containerização **Docker** multi-stage e código seguindo clean code (SOLID, type hints, Factory + Strategy).

> Status: em desenvolvimento — Etapa 1 (estrutura e clean code) concluída.

## Estrutura do projeto

```
├── configs/            # Configurações (Pydantic Settings)
├── data/               # Dados (versionados via DVC, fora do Git)
│   ├── raw/
│   └── processed/
├── models/             # Artefatos de modelo (DVC)
├── scripts/            # validate_env.py, promote_model.py etc.
├── src/recsys/
│   ├── data/           # Loaders e preprocessors (Strategy)
│   ├── features/       # Engenharia de features
│   ├── models/         # ModelFactory (Factory), NCF, baselines
│   ├── training/       # Loop de treino, early stopping, MLflow
│   └── evaluation/     # Recall@K, Precision@K, NDCG@K, HitRate@K
└── tests/              # Testes unitários (pytest)
```

## Instalação (macOS e Windows)

Pré-requisitos: [uv](https://docs.astral.sh/uv/) instalado.

```bash
git clone https://github.com/rafaneder-maiorino/recsys-ecommerce.git
cd recsys-ecommerce
uv sync            # cria .venv e instala deps exatas do uv.lock
uv run pytest      # valida a instalação
uv run ruff check  # linting
```

> Equivalência Poetry: `uv sync` cumpre o mesmo papel de `poetry install` — instalação determinística a partir do lock file (`uv.lock`), conforme Aula 02 de Gerenciamento de Dependências.

## Qualidade de código

- `ruff` (lint + format) com convenção de docstrings Google e limite de complexidade
- `pre-commit` com hooks de ruff e bloqueio de arquivos grandes (dados vão pelo DVC)

```bash
uv run pre-commit install
```

## Design patterns aplicados

- **Factory Method** (`src/recsys/models/factory.py`) — instancia modelos por nome vindo do `params.yaml`, sem `if/else` espalhado
- **Strategy** (`src/recsys/data/preprocessors.py`) — etapas de pré-processamento intercambiáveis e composáveis

## Roadmap

Ver [`docs/roteiro-tech-challenge.md`](docs/roteiro-tech-challenge.md). Próximas etapas: DVC + dados (Etapa 2), pipeline + Docker (Etapa 3), NCF + Registry (Etapa 4), deploy bônus no Azure.
