# Roteiro — Tech Challenge Fase 2 (FIAP Pós Tech MLE)

**Projeto:** `recsys-ecommerce` — sistema de recomendação para e-commerce
**Prazo de entrega:** 14/07/2026 · **Meta:** nota 9–10
**Entrega obrigatória:** Repositório GitHub + vídeo STAR ≤ 5 min · **Bônus:** deploy em nuvem (5%)

---

## Princípio orientador

A rubrica concentra **70% da nota em engenharia** (clean code 15% + reprodutibilidade 15% + Docker 15% + DVC 15% + MLflow 10%) e apenas 15% no modelo. Regra do projeto: **nenhum dia termina sem o repositório em estado "entregável"**. Commits semânticos (`feat:`, `fix:`, `docs:`, `chore:`, `test:`) desde o primeiro commit — o histórico é critério avaliado e não pode ser reescrito depois.

## Decisões de arquitetura (fechadas)

| Decisão | Escolha | Justificativa |
|---|---|---|
| Dataset | **RetailRocket** (~2,7M eventos) | Enunciado pede "comportamento de navegação"; eventos view/addtocart/transaction = feedback implícito real de e-commerce |
| Modelo | **NCF** (embeddings user/item + MLP) | Satisfaz "MLP ou embedding-based" simultaneamente; padrão da literatura; treina no M2 (MPS) |
| Baselines | Popularidade, ItemKNN, TruncatedSVD (sklearn) | ≥ 3 runs MLflow exigidos; comparação exigida pela rubrica |
| Métricas | Recall@10, Precision@10, NDCG@10, HitRate@10 (+MAP) | ≥ 4 métricas exigidas; canônicas de ranking |
| Split | **Temporal** (leave-last-out) | Split aleatório vaza futuro em recomendação — ponto para o vídeo |
| Gerenciador | **uv** + `uv.lock` | Aula 02 de Dependências: "escolha mais moderna para MLOps"; README documenta equivalência com `poetry install` |
| Patterns | **Factory Method** (modelos) + **Strategy** (preprocessors) | Exatamente os sugeridos no enunciado e na Aula 03 de Clean Code |
| DVC remote | **Azure Blob Storage** + SAS token de leitura | Aula 03 DVC cita Azure Blob; sinergia com deploy bônus; avaliador roda `dvc pull` sem conta |
| Registry | Stages clássicos None→Staging→Production + tags de governança | Fluxo do Hands On da Aula 05; tags `approved_by`/`approval_notes` + quality gate como diferencial |
| Docker | Multi-stage `python:3.11-slim`, non-root, torch CPU-only | Template da Aula 03 de Docker; hardening = diferencial barato |

## Padrões de integração DVC+MLflow (Aula 06)

1. `params.yaml` = fonte única da verdade, segmentado por stage, declarado no `dvc.yaml` e logado via `mlflow.log_params()`
2. Métricas duplas: `metrics.json` como `metrics:` no dvc.yaml (`dvc metrics show/diff`) **e** `mlflow.log_metric()`
3. Tag `train_data_version` no MLflow com hash DVC do dataset (rastreabilidade dados↔modelo)

---

## Cronograma (6 dias + buffer)

### ✅ Dia 1 — Qua 08/07 · Etapa 1 FIAP: Clean Code e Estrutura (15%)

- [x] Estrutura `src/`, `tests/`, `data/`, `models/`, `configs/`, `scripts/`
- [x] `pyproject.toml` (uv) com deps prod/dev separadas
- [x] Ruff (convenção Google, max-complexity 8) + pre-commit hooks
- [x] `ModelFactory` (Factory Method) + `PreprocessingStrategy` (Strategy) com type hints e docstrings Google
- [x] `.gitignore`, `.dockerignore`, `.env.example`
- [x] Testes unitários dos patterns
- [ ] `uv sync` + `uv.lock` commitado (executar no Mac)
- [ ] Repo GitHub criado, commits semânticos

**DoD:** `uv sync` limpo · `ruff check` zero erros · `pytest` verde · ~8-12 commits semânticos

### Dia 2 — Qui 09/07 · Etapa 2 FIAP: Ambiente + dados (15%)

- [ ] `configs/settings.py` com Pydantic Settings lendo `.env`
- [ ] `scripts/validate_env.py` (Python, libs, MLflow, variáveis)
- [ ] Download RetailRocket + EDA mínima (distribuição de eventos, esparsidade)
- [ ] Decisões de dados: filtro min-interações, split temporal, pesos por evento
- [ ] `dvc init` + remote Azure Blob + dataset versionado + `dvc push`
- [ ] Seeds globais fixados via config

**DoD:** clone limpo → `uv sync` → `validate_env.py` OK → `dvc pull` traz dados

### Dia 3 — Sex 10/07 · Etapa 3 FIAP: Pipeline + Docker (30%)

- [ ] `dvc.yaml` com 4 stages: `preprocess → feature_eng → train → evaluate`
- [ ] `params.yaml` segmentado por stage
- [ ] `metrics.json` como `metrics:` + log no MLflow + tag `train_data_version`
- [ ] `Dockerfile` multi-stage (builder+runtime, slim, non-root, healthcheck, torch CPU)
- [ ] `docker-compose.yml`: serviço `mlflow-server` + serviço `train` (roda `dvc repro`)
- [ ] Modelo trivial (popularidade) fechando o pipeline de ponta a ponta

**DoD:** `dvc repro` reproduzível · `docker compose up` funcional · run na UI do MLflow

### Dia 4 — Sáb 11/07 · Etapa 4a: Baselines + NCF (15%)

- [ ] Baselines sklearn via Factory: Popularidade, ItemKNN, TruncatedSVD
- [ ] Módulo `evaluation/` com as 5 métricas + protocolo leave-last-out
- [ ] NCF PyTorch: embeddings → MLP → sigmoid, negative sampling, **early stopping** (nominal na rubrica)
- [ ] Testes unitários de métricas e split
- [ ] ≥ 3 runs completos no MLflow

**DoD:** NCF supera baselines em NDCG@10 e Recall@10

### Dia 5 — Dom 12/07 · Etapa 4b: Registry + docs (10% + sustenta 15%)

- [ ] `scripts/promote_model.py`: registro + **quality gate** + Staging→Production + tags `approved_by`/`approval_notes`
- [ ] `MODEL_CARD.md`: performance, limitações (cold start, viés de popularidade), vieses, uso pretendido
- [ ] README final PT-BR: arquitetura, instruções Mac+Windows, screenshots MLflow
- [ ] **Teste do avaliador no Mac**: clone do zero, seguir README literalmente

### Dia 6 — Seg 13/07 · Bônus Azure + Windows + vídeo (5% + 10%)

- [ ] API FastAPI (`/recommend/{user_id}`, `/health`) carregando modelo `Production` do Registry
- [ ] Deploy container no Azure (Container Apps/App Service) + URL pública no README
- [ ] **Teste de clone limpo no Windows**
- [ ] Vídeo STAR ≤ 5 min:
  - Situation (~45s): e-commerce, RetailRocket, feedback implícito
  - Task (~45s): requisitos técnicos e restrições
  - Action (~2min): arquitetura, por que NCF, split temporal, uv, DVC+MLflow+Docker, trade-offs
  - Result (~1,5min): métricas vs baselines, modelo em Production, URL do deploy, lições

### Dia 7 — Ter 14/07 · Buffer e entrega

- [ ] Revisão final do histórico de commits
- [ ] Upload do vídeo + submissão (manhã, nunca no último minuto)

---

## Riscos e mitigações

1. **RetailRocket grande (~2,7M eventos)** → amostragem configurável no `params.yaml` (500k–1M interações); documentar como decisão consciente
2. **Imagem Docker pesada com PyTorch** → wheel CPU-only (`--index-url .../cpu`), corta ~5GB; mencionar no vídeo como otimização
3. **Dia 3 estourar** → modelo trivial garante pipeline funcional independente do NCF

## Mapa rubrica → entregável

| Critério | Peso | Onde é atendido |
|---|---|---|
| Clean code e estrutura | 15% | Dia 1: SOLID, patterns, ruff, type hints |
| Reprodutibilidade | 15% | Dias 1-2 + 5: uv.lock, .env, validate_env, testes de clone Mac/Windows |
| Docker | 15% | Dia 3: multi-stage, non-root, compose |
| DVC + Pipeline | 15% | Dias 2-3: remote Azure, 4 stages, dvc repro |
| Rede neural | 15% | Dia 4: NCF, early stopping, baselines |
| MLflow + Registry | 10% | Dias 3-5: ≥3 runs, quality gate, Production |
| Vídeo STAR | 10% | Dia 6 |
| Bônus nuvem | 5% | Dia 6: URL pública Azure |
