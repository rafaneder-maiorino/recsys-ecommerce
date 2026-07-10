# Model Card — recsys-ncf

Neural Collaborative Filtering para recomendação de produtos em e-commerce, treinado sobre o dataset RetailRocket. Estrutura baseada em Mitchell et al. (2019), *Model Cards for Model Reporting*.

## Detalhes do modelo

| Campo | Valor |
|---|---|
| Arquitetura | NCF (He et al., 2017): embeddings de usuário e item (dim 32) concatenados → MLP [64, 32] com dropout 0.2 → logit |
| Framework | PyTorch 2.13 (treino em Apple MPS; inferência em CPU) |
| Objetivo de treino | BCE com logits sobre positivos implícitos + 4 negativos amostrados por positivo; pesos de evento (view=1, addtocart=3, transaction=5) escalam a loss dos positivos |
| Regularização | Dropout + **early stopping** (paciência 3 sobre a val loss; disparou na época 24, pesos restaurados da época 21) |
| Seleção | Melhor val loss = 0.3221 (época 21) |
| Registro | MLflow Model Registry `recsys-ncf`, promovido a **Production** via quality gate |

## Uso pretendido

Recomendação top-k de produtos com base no histórico de navegação (feedback implícito). Uso primário: ranquear candidatos para usuários **conhecidos** (presentes no treino). Não se destina a decisões sensíveis sobre pessoas.

## Dados de treinamento

- **Fonte:** [RetailRocket](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset) — 2.756.101 eventos reais de e-commerce (2,66M views, 69k add-to-carts, 22k transações), 1.407.580 visitantes, 235.061 itens, janela de 137 dias (mai–set/2015), valores hasheados por confidencialidade.
- **Pré-processamento:** filtro 5-core (usuários com ≥ 5 interações) → 948.537 interações, 81.620 usuários, 103.873 itens; densidade da matriz ~0,008%.
- **Split temporal (leave-last-out):** última interação de cada usuário → teste; penúltima → validação; restante → treino. Split aleatório vazaria informação futura.
- **Rastreabilidade:** cada run do MLflow carrega a tag `train_data_version` com o hash DVC do snapshot exato dos dados.

## Protocolo de avaliação

Ranking com candidatos amostrados (protocolo do paper do NCF): o item verdadeiro de cada usuário de teste é ranqueado contra **99 itens nunca vistos** pelo usuário (amostragem com seed fixa). Justificativa: ranquear o catálogo completo (103.873 itens × 76.404 usuários de teste) exigiria ~8 bilhões de forward passes do MLP; o protocolo amostrado mantém a comparação idêntica entre todos os modelos. Sob leave-last-out com um único item relevante, Recall@10 ≡ HitRate@10.

## Resultados (test set, 76.404 usuários)

| Modelo | HitRate@10 | Precision@10 | NDCG@10 | MRR@10 |
|---|---|---|---|---|
| Popularity (baseline) | 0.541 | 0.054 | 0.320 | 0.253 |
| ItemKNN | **0.949** | **0.095** | **0.810** | **0.766** |
| TruncatedSVD (linear, 64 fatores) | 0.541 | 0.054 | 0.373 | 0.322 |
| **NCF (early stopping @ época 24)** | 0.667 | 0.067 | 0.474 | 0.414 |

**Leitura dos resultados.** O NCF supera com folga os dois comparativos "justos": +23% de HitRate sobre popularidade e SVD, e +27% de NDCG sobre o SVD — sendo o SVD a fatoração *linear*, essa diferença isola o ganho da não-linearidade do MLP. O ItemKNN lidera o benchmark, e entender o porquê é essencial: no RetailRocket, a última interação do usuário é frequentemente um item que ele **já havia tocado antes** (funil view → addtocart → transaction do mesmo produto). O item removido do treino permanece no histórico usado pelo KNN, cuja auto-similaridade domina o score — ou seja, o KNN vence em grande parte prevendo **reconsumo**, enquanto o NCF generaliza para interações novas. Métodos de vizinhança serem baselines extremamente fortes em next-item de e-commerce é resultado recorrente na literatura.

## Limitações

1. **Cold start total:** usuários e itens ausentes do treino não têm embedding; itens frios foram removidos da avaliação (77.477 → 76.404 linhas de teste).
2. **Viés de popularidade:** feedback implícito super-representa itens populares; o negative sampling uniforme mitiga apenas parcialmente.
3. **Janela temporal curta:** 4,5 meses de dados; sazonalidade anual não é capturada e o modelo requer retreino periódico.
4. **Reconsumo:** o protocolo atual não separa "recomendar algo novo" de "prever repetição" — a comparação com o ItemKNN mistura as duas tarefas (ver leitura acima).
5. **Sem features de conteúdo:** IDs hasheados impedem uso de atributos de produto; o modelo é puramente colaborativo.

## Considerações éticas

Dados anonimizados na origem (IDs hasheados). O modelo pode reforçar viés de popularidade, reduzindo a exposição de itens de cauda longa; mitigação (re-ranking com diversidade) fica fora do escopo desta fase.

## Reprodução

```bash
git clone https://github.com/rafaneder-maiorino/recsys-ecommerce.git
cd recsys-ecommerce && uv sync
dvc pull            # requer token de leitura (ver README)
docker compose up -d mlflow
uv run dvc repro    # pipeline completo, seed fixa = 42
uv run python scripts/promote_model.py
```
