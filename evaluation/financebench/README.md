# Évaluation FinanceBench

Évaluation du RAG agentique sur des filings SEC longs et denses en tableaux — le seul jeu
d'éval du projet, et le seul qui porte un chiffre publiable (annotation experte externe).

## Pourquoi FinanceBench

[FinanceBench](https://github.com/patronus-ai/financebench) (Islam et al., 2023, Patronus AI) est
le benchmark que **Mistral a utilisé pour évaluer son produit Agentic Search**
([annonce](https://mistral.ai/news/agentic-search/)) : QA financière sur des filings SEC (10-K,
10-Q, 8-K) longs et denses en tableaux — ~147 pages par document en moyenne. Mistral y annonce
un passage de **26,7 % à 86 %** de correctness avec Agentic Search, jugé par un LLM calibré sur
des annotations humaines.

Ce que le benchmark met en difficulté et que les documents précédents ne testaient pas :

- **Densité** : un 10-K fait 150 à 260 pages, contre ~30 pages pour les documents précédents.
- **Tableaux** : les réponses se trouvent dans des états financiers consolidés, pas dans du texte
  narratif — l'OCR et le chunking doivent les préserver.
- **Distracteurs** : chaque chiffre apparaît des dizaines de fois dans le document (exercices
  différents, segments différents, notes annexes).
- **Questions à réponse négative** : plusieurs questions attendent « il n'y en a pas » ou
  « cette métrique ne s'applique pas à cette entreprise » — un piège à hallucination.

> ⚠️ À ne pas confondre avec [`yuweiyin/FinBench`](https://huggingface.co/datasets/yuweiyin/FinBench),
> un dataset de **classification tabulaire** (défaut / fraude / churn) qui ne contient aucune
> question et n'est pas un benchmark de RAG.

## Sous-ensemble évalué

Le corpus complet fait 705 Mo (368 PDF, ~53 900 pages). On évalue les **3 documents portant
7 questions chacun**, plus PepsiCo pour un signal numérique exact (réponses de type `$9068.00`),
soit **26 questions** :

| Document | Questions | PDF | Pages |
|---|---|---|---|
| `AMD_2022_10K` | 7 | 5,1 Mo | ~120 |
| `AMERICANEXPRESS_2022_10K` | 7 | 2,4 Mo | 260 |
| `BOEING_2022_10K` | 7 | 1,4 Mo | ~150 |
| `PEPSICO_2022_10K` | 5 | | 503 |

Répartition : 17 `domain-relevant` + 7 `novel-generated` + 2 `metrics-generated`, mêlant
extraction d'information, raisonnement numérique et raisonnement logique.

Les 4 documents sont indexés **dans un seul index** : le système doit retrouver le bon passage
du bon document, sans savoir lequel porte la réponse. `--per-doc` donne le réglage plus
facile (un index par document).

Les runs antérieurs au 4 septembre 2026 (après-midi) portaient sur les 3 premiers documents,
21 questions : leurs chiffres ne sont pas comparables à ceux d'un run à 26.

### Jeu étendu (70 questions)

À 26 questions, l'IC95 de l'accuracy fait ~35 points de large : un écart de 5 points entre deux
versions n'y est pas mesurable. `--docs extended` prend tous les documents qui portent au moins
3 questions : **18 documents, 70 questions**. Il s'écrit dans `dataset_extended.jsonl`, jamais
dans `dataset.jsonl` (le protocole publié), et le runner refuse d'en écrire les résultats dans
`outputs/` :

```bash
uv run python evaluation/financebench/prepare.py --docs extended
uv run python evaluation/financebench/run_financebench_eval.py --mode both \
    --dataset evaluation/financebench/dataset_extended.jsonl \
    --out-dir evaluation/financebench/outputs_dataset_extended
```

La préparation OCRise 14 documents de plus. Coût estimé, pas mesuré : de l'ordre de 2 000 pages,
soit ~8 $ d'OCR à la grille de `cost.py`. Un run complet coûte environ 70/26 fois celui du
protocole à 26 questions.

## Prérequis

Un fichier `.env` à la racine du projet :

```bash
MISTRALAI_API_KEY=...
COHERE_API_KEY=...
LANGSMITH_API_KEY=...   # optionnel
```

## Lancer l'évaluation

L'évaluation se fait en deux temps, parce que le coût dominant est l'ingestion (~1 070 pages à
OCRiser), pas les questions.

```bash
# 1) Préparation — une seule fois (~5-15 min). Télécharge, OCRise, chunke, embed, met en cache.
uv run python evaluation/financebench/prepare.py

# 2) Le run évalué — cible 5-10 min, reproductible
uv run python evaluation/financebench/run_financebench_eval.py --mode both
```

La préparation met en cache l'OCR (`cache/*.ocr.json`), les chunks (`cache/*.chunks.pkl`) et les
embeddings (collection Chroma persistée dans `chroma/`). Les runs suivants repartent du cache.
Seuls `dataset.jsonl` et ce README sont versionnés ; PDF, caches et résultats sont gitignorés.

### Options utiles

| Flag | Effet |
|---|---|
| `--mode baseline\|agentic\|both` | Défaut `both` (mesure l'apport de la boucle agentique) |
| `--workers N` | Questions en parallèle (défaut 3). **Passer à 1 avec une clé Cohere trial** (~10 req/min) |
| `--max-items N` | Limite le nombre de questions (test rapide) |
| `--no-judge` | Désactive le juge LLM (pas de métriques d'accuracy) |
| `--time-budget S` | Arrêt propre au-delà de S secondes, avec résultats partiels (défaut 600) |
| `--per-doc` | Un index par document au lieu de l'index combiné |
| `--page-tolerance N` | Écart de pagination toléré entre PDF annoté et OCR (défaut 1) |
| `--verbose` | N'étouffe pas les logs de debug du pipeline |

### Si l'ingestion tape un rate limit (HTTP 429)

L'OCR d'un 10-K de 260 pages consomme beaucoup de quota d'un coup. `prepare.py` rejoue
automatiquement avec un backoff exponentiel (jusqu'à 7 tentatives, en respectant l'en-tête
`Retry-After`), découpe l'OCR en lots de 50 pages et **écrit le cache après chaque lot** :
une interruption ne fait jamais perdre des pages déjà payées, et relancer la commande reprend
exactement là où ça s'était arrêté.

Si les 429 persistent, réduisez la pression :

```bash
uv run python evaluation/financebench/prepare.py --ocr-batch 20 --ocr-delay 10
```

Et si besoin, ingérez un document à la fois :

```bash
uv run python evaluation/financebench/prepare.py --docs AMD_2022_10K
uv run python evaluation/financebench/prepare.py --docs AMERICANEXPRESS_2022_10K
uv run python evaluation/financebench/prepare.py --docs BOEING_2022_10K
uv run python evaluation/financebench/prepare.py --docs PEPSICO_2022_10K
uv run python evaluation/financebench/prepare.py   # assemble et construit l'index
```

## Métriques

### Protocole FinanceBench (comparable au chiffre public de Mistral)

Chaque réponse reçoit un verdict ternaire d'un juge LLM :

- **`accuracy`** — réponses `CORRECT` (tolérante aux écarts de format, d'unité et d'arrondi).
- **`hallucination_rate`** — réponses `INCORRECT` : le système répond avec assurance **et** se
  trompe. C'est la métrique centrale du papier — le risque réel en usage professionnel.
- **`refusal_rate`** — réponses `REFUSAL` : le système déclare ne pas savoir. Un refus n'est pas
  une bonne réponse, mais il n'est pas dangereux.

Chaque taux est publié avec son **comptage brut** (`counts`) et son **intervalle de confiance à
95 %** (Wilson, `*_ci95`), et le tableau de synthèse les affiche.

> ⚠️ **26 questions, c'est peu.** L'IC95 d'une accuracy autour de 80 % fait environ **30 points de
> large** : 20/26 et 22/26 ne sont pas distinguables. Une différence d'une ou deux questions entre
> deux configurations n'est pas une amélioration, c'est du bruit — le run l'écrit explicitement en
> fin de rapport. Ne citer un gain que s'il tient sur plusieurs questions **et** que les intervalles
> ne se recouvrent quasiment plus.

Pour recalculer un intervalle à la main, ou l'obtenir depuis un résumé déjà écrit :

```bash
uv run python evaluation/financebench/confidence.py 22 26
uv run python evaluation/financebench/confidence.py --summary evaluation/financebench/outputs/financebench_summary.json
```

Les refus sont détectés **sans appel LLM** quand le pipeline renvoie un de ses messages figés
(`backend/agents/research_agent.py`, `backend/agents/workflow.py`), ce qui économise du budget.
Les erreurs techniques (timeout, LLM indisponible) sont comptées à part et exclues du
dénominateur, pour ne pas les confondre avec des refus.

### Faithfulness et answer relevancy

- **`mean_faithfulness`** (1-5) — note donnée par le juge dans le même appel que le verdict : la
  réponse s'en tient-elle aux extraits fournis ? Ce n'est pas la faithfulness de RAGAS
  (vérification affirmation par affirmation), mais un signal global.
- **`mean_answer_relevancy`** (0-1) — la réponse traite-t-elle la question posée ? Méthode RAGAS
  (`evaluation/answer_relevancy.py`) : Mistral Small écrit 3 questions auxquelles la réponse
  répond, et on prend la similarité cosinus moyenne (Mistral Embed) avec la question d'origine.
  Un refus ou une réponse évasive vaut 0. Elle ne regarde pas la référence : une réponse fausse
  mais centrée sur la question score haut ; c'est le verdict du juge qui dit si elle est juste.
  Coût : un appel Mistral Small par réponse. `--no-answer-relevancy` la désactive.

Sur les réponses sauvegardées du run du 4 septembre (calcul a posteriori, même code) :

| | Faithfulness /5 | Answer relevancy | … sur les CORRECT | … sur les INCORRECT | … sur les refus |
|---|---|---|---|---|---|
| Baseline | 4,73 | 0,710 (26) | 0,840 (17) | 0,596 (7) | 0,0 (2) |
| Agentic | 4,79 | 0,805 (24) | 0,798 (20) | 0,842 (4) | — |

L'écart de relevancy entre les modes vient surtout des refus, qui valent 0 : l'agentic n'en a
pas. Les 4 hallucinations de l'agentic restent centrées sur la question (0,84) : elles se
trompent de chiffre, elles ne répondent pas à côté.

### Retrieval exact, grâce aux pages annotées

FinanceBench annote la **page** de chaque preuve (zero-indexed). L'ingestion conserve le numéro de
page en metadata sur chaque chunk, ce qui donne des métriques de retrieval sans matching flou :

- **`page_hit@k`** — au moins un des k premiers chunks vient-il d'une page de preuve ?
- **`page_recall@k`** — quelle proportion des pages de preuve distinctes est couverte ?

Une tolérance de ±1 page absorbe un éventuel décalage entre la pagination du PDF et celle de l'OCR.

S'y ajoutent, par mode :

- **`evidence_seen_rate`** — la preuve figure-t-elle parmi les documents effectivement transmis au
  LLM ? C'est la métrique qui borne l'accuracy : le modèle ne peut pas répondre juste à partir
  d'une preuve qu'il n'a pas vue. Pour le mode agentic, elle compte aussi les passages ramenés
  par les outils ; le delta avec baseline mesure donc l'apport réel de la boucle agentique.
- **`corrective_rate`** (agentic) — part des questions où le modèle a appelé au moins un outil.
- **`mean_tool_calls_when_corrected`** / **`pages_read_rate_when_corrected`** (agentic) — sur ces
  questions, nombre moyen d'appels d'outils, et part où au moins une page entière a été lue
  (`read_page`). Chaque ligne de résultat liste aussi `pages_read` et la trace des appels dans
  `corrective_queries`.
- `financebench_results.json` contient aussi, par question, les 20 `(document, page)` récupérés
  et `gold_rank` (rang de la première page de preuve), pour diagnostiquer le retrieval sans
  relancer d'appels API.

### Métriques de retrieval

`recall@k`, `precision@k`, `mrr@k`, `ndcg@k` (matching textuel sur les passages de preuve),
`context_hit_rate` et `mean_f1` viennent de `evaluation/metrics.py`. `page_precision@k` est la
part des k premiers passages qui viennent d'une page de preuve.

⚠️ **La précision est plafonnée par construction.** Une question n'a souvent qu'une ou deux pages
de preuve : même un retrieval parfait ne met pas 10 passages pertinents dans un top-10. Elle se
lit d'un run à l'autre (le contexte contient-il moins de bruit ?), pas en valeur absolue. Run du
4 septembre, recalculé depuis les pages sauvegardées : `page_precision@5` 14,6 %,
`@10` 13,1 %, `@20` 8,7 % (identique dans les deux modes : même retrieval initial).

⚠️ **`mean_f1` est peu informatif ici** : les réponses attendues sont en prose
(« Data Center », « Performance is not measured through operating margin »), le recouvrement de
tokens ne mesure donc pas la justesse. C'est bien l'`accuracy` du juge qui fait foi — comme dans
le protocole officiel.

Le résumé est ventilé par `question_type` et par `question_reasoning`, pour voir sur quel type de
raisonnement le système décroche.

## Sorties

Dans `evaluation/financebench/outputs/` (**versionnés** : ce sont les chiffres cités dans le
README et l'étude de cas, ils doivent être vérifiables sans relancer l'éval).

> Un run partiel (`--max-items`, `--no-judge`, `--docs`) **refuse** d'écrire dans ce répertoire :
> il remplacerait les chiffres publiés par un échantillon de trois questions sans verdict. Utilisez
> `--out-dir /tmp/...` pour un essai, `--force-overwrite` si le remplacement est voulu.


- `financebench_summary.json` — métriques agrégées par mode, deltas, ventilations, et le bloc
  `cost` : tokens facturés par modèle (callback LangChain sur chaque appel, juge compris), unités
  de recherche Cohere (`billed_units` de chaque réponse Rerank), total en dollars et en euros à
  la grille publique du jour (`evaluation/financebench/cost.py`)
- `financebench_results.json` — le détail par question (réponse, verdict, raison du juge, latences)
- `financebench_errors.json` — questions ayant échoué techniquement

Un tableau de synthèse est aussi affiché en fin de run.

## Coût d'un run

Le tableau de fin de run affiche le coût, calculé à partir des tokens réellement facturés et de
la grille publique de La Plateforme et de Cohere (`cost.py`, avec la date de relevé et le taux de
change utilisés). Ce n'est pas une estimation : ce sont les `usage` renvoyés par chaque appel.

Run complet du 5 septembre 2026 (26 questions, `--mode both`, juge compris, 260 s) :

| Poste | Volume | Coût |
|---|---|---|
| `mistral-large-latest` (génération des deux modes + juge) | 421 872 tokens en entrée, 29 129 en sortie | 0,25 $ |
| `mistral-small-latest` (sous-agents) | 29 862 en entrée, 137 en sortie | 0,005 $ |
| Cohere `rerank-v4.0-pro` | 34 recherches (une par appel, ≤ 40 documents de ~300 tokens) | 0,09 $ |
| **Total** | | **≈ 0,35 $ ≈ 0,30 €** |

Les tokens Mistral sont ceux facturés (`usage` de chaque réponse). Les recherches Cohere sont
comptées au tarif public par appel tenté : pendant cette mesure, la clé Cohere avait atteint son
plafond mensuel de facturation (HTTP 402) et le reranker a rendu l'ordre du retriever hybride —
le run du 4 septembre, celui des chiffres publiés, a subi le même plafond sur ses 14 derniers
appels Rerank seulement (192 réussis avant).

Ne sont pas comptés les embeddings de requête pendant le run (quelques dizaines de tokens par
recherche, négligeables) et la préparation, payée une fois : l'OCR des 1 074 pages et l'embedding
des ~12 400 chunks, soit 4,4 $ ≈ 3,8 € (OCR à 4 $ les 1 000 pages, ~900 k tokens embeddés à 0,10 $/M) à la même grille.

## Garde-fous de régression

L'éval complète mesure le système ; elle ne dit pas *ce qui* a cassé et coûte trop pour tourner
à chaque changement. `regression.py` fournit deux contrôles ciblés, à lancer **à la main** avant
de merger un changement de prompt, de chunking, de retrieval ou de modèle. Ils ne tournent pas en
CI : ils appellent les vraies API. La CI, elle, fait tourner les tests pytest, tous hors ligne.

```bash
# Retrieval seul, question par question contre une baseline versionnée (quelques centimes)
uv run python evaluation/financebench/regression.py retrieval

# ~10 questions sentinelles, de bout en bout avec le juge (~0,10 €)
uv run python evaluation/financebench/regression.py sentinels
```

**`retrieval`** rejoue le retrieval des 26 questions et compare le rang de la page de preuve à
`regression/retrieval_baseline.json`. Il échoue si une question perd sa preuve du top-10 (ce
que le modèle voit sans outils) ou si `page_hit@10` baisse de plus de 5 points. La comparaison
est faite question par question, parce qu'une perte compensée par un gain laisse la moyenne
inchangée. Après un changement voulu qui améliore le retrieval :
`regression.py retrieval --update-baseline`.

**`sentinels`** rejoue en mode agentic les questions de `regression/sentinels.json`, réussies au
run du 4 septembre, et vérifie le verdict du juge **et** la trajectoire : la page de preuve
a-t-elle été montrée au modèle ? Chaque sentinelle dit ce qu'elle protège (calcul de ratio,
preuve hors du top-10 atteinte seulement par les outils, réponse directe…). Un échec est
toléré par défaut (`--max-failures 1`) : le modèle à température 0 n'est pas strictement
déterministe ; relancer la question en échec départage.

Codes de sortie : `0` passe, `1` régression, `2` **run invalide**. Quand Cohere échoue (quota,
clé), le reranker rend l'ordre du retriever hybride sans lever d'erreur : le retrieval baisse et
ressemblerait à une régression du code. Les deux garde-fous le détectent et refusent de conclure.

La baseline actuelle est reprise du run versionné du 4 septembre, dont les derniers appels
Rerank avaient atteint le plafond Cohere : elle est à régénérer par un run live dès que la clé
fonctionne.

## Notes d'implémentation

- **Chunking récursif, pas sémantique.** La stratégie sémantique
  (`SemanticParentChildChunkingStrategy`) embed les paragraphes par batchs de 16 en série : sur
  un 10-K de 260 pages, cela représente des centaines d'appels API séquentiels. La préparation
  utilise donc explicitement `ParentChildChunkingStrategy` (parents 1200 / children 400).
- **Un seul retrieval par question.** `AgentWorkflow.full_pipeline()` relance un retrieval
  complet ; le runner invoque directement le graphe compilé avec les documents déjà récupérés,
  ce qui divise la latence par ~2 sans changer le comportement mesuré.
- **Comparaison honnête.** Les deux modes partagent exactement le même retrieval et les mêmes
  10 documents de contexte initiaux (`workflow._research_step`). Le seul écart est la boucle agentique :
  le delta mesure donc son apport propre, pas une différence de contexte.
- **Index scopé au jeu de documents.** Le répertoire Chroma est nommé d'après un hash de la liste
  de documents, pour éviter qu'un run lancé avec `--docs` sur un sous-ensemble n'interroge une
  collection contenant d'autres documents.

## Licence

FinanceBench est distribué par Patronus AI (CC-BY-NC-4.0). Les PDF ne sont pas redistribués dans
ce dépôt : `prepare.py` les télécharge depuis le dépôt public.
