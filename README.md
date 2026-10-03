# RAG Agentique évalué ~96 % de réponses correctes sur un sous-ensemble de FinanceBench
![RAG Agentique multi-agent Header](./static/header-c.webp)

Si vous appréciez, ajoutez une ⭐ au repo pour soutenir mon travail. 🙏

Ce système RAG combine un récupérateur hybride (BM25 + embeddings dans Qdrant, reranking Cohere), un routage
par document et un modèle de réponse équipé d'outils (`search` / `grep` / `read_page` / `open_document` / `navigate`), sur des
rapports SEC de 150 à 260 pages. Il est **mesuré** sur
[FinanceBench](https://github.com/patronus-ai/financebench), le benchmark utilisé par Mistral pour
évaluer Agentic Search (150 questions). Le résultat qui compte est l'ablation, à retrieval
strictement identique : **96,2 % de réponses correctes avec les outils, contre 76,9 % sans**, et
des hallucinations qui passent de 15,4 % à 3,8 %. Dix-neuf points gagnés par l'agent équipé, sur
le même index et les mêmes 10 passages initiaux. Le run complet coûte environ 1 € à relancer.
Tous les chiffres sont reproductibles à partir des sorties dans
`evaluation/financebench/outputs_sonnet5/` —
[résultats, coût et limites](#évaluation-financebench-documents-financiers-difficiles).

## Architecture IA à la base avant améliorations

![Projet Overview](./static/project-overview.jpg)

### 1. **Agent Vérificateur de Pertinence**
Évalue si les passages récupérés répondent réellement à la question (CAN_ANSWER / PARTIAL / NO_MATCH). Son verdict ne bloque plus la génération : il est transmis au modèle de réponse comme indice (« le contexte initial a été jugé partiel, cherchez ce qui manque »).

### 2. **Agent de Recherche et de réponse, avec outils**
Le modèle de génération reçoit les 10 meilleurs passages et les [cinq outils](#les-outils) décrits plus bas. Il répond directement si le contexte suffit ; sinon il cherche — en voyant chaque résultat avant de décider du suivant, 5 appels au plus — et répond **dans la même conversation** : le modèle qui cherche est celui qui répond, comme dans l'[Agentic Search](https://mistral.ai/news/agentic-search/) de Mistral. Les outils sont disponibles sur **toutes** les questions ; le mode conditionnel (agent de recherche séparé, ou réécriture de la question) reste disponible pour comparaison via `GENERATOR_TOOLS_ENABLED` et `CORRECTIVE_MODE`.

### 3. **Génération contrainte**
La réponse ne s'appuie que sur les passages numérotés — initiaux ou ramenés par les outils — avec une citation `[n]` après chaque affirmation, et refuse quand l'information n'y est pas. Deux règles de prompt tirées des runs FinanceBench : un ratio ou une marge dont les composantes sont dans le contexte se **calcule** (formule, chiffres cités, résultat) ; et « non disponible » ne s'écrit qu'après un `grep` sans résultat.

## Cet agent a des outils à dispo

Cinq opérations façon système de fichiers — les cinq de l'[Agentic Search de Mistral](https://mistral.ai/fr/news/agentic-search/) (search, open, navigate, read, grep) —, données au modèle de réponse. Les pages OCR sont conservées entières (`backend/retriever/page_store.py`) à côté des chunks : les chunks servent à *trouver*, les pages à *lire*.

| Outil | Ce qu'il fait |
|---|---|
| `search(query, doc)` | Le retrieval hybride du pipeline (BM25 + vecteurs dans Qdrant, routage, rerank Cohere), relancé avec une nouvelle requête, optionnellement restreint à un document. Renvoie 8 extraits avec document et page. |
| `grep(pattern, doc)` | Occurrences littérales d'un motif (regex, insensible à la casse), page par page, sur tout le document. Exhaustif : 0 résultat permet d'affirmer qu'un terme n'y figure pas. |
| `read_page(doc, page, end_page)` | La page entière telle que l'OCR l'a produite, tableau compris — ce qu'un chunk de 1 200 caractères ne montre jamais. `end_page` lit 2 à 3 pages d'un coup pour un tableau à cheval. |
| `open_document(doc)` | Le plan du document — PART, ITEM, états financiers consolidés, notes — avec la page de chaque section, et son nombre de pages. Sans `doc`, la liste des documents. Les niveaux d'en-têtes de l'OCR n'étant pas fiables (200 à 550 titres par 10-K), le plan retient les repères structurels par leur intitulé. |
| `navigate(doc, section)` | La page où commence une section, cherchée dans les **titres** seulement (« consolidated balance sheet », « income taxes ») : là où `grep` renvoie aussi le sommaire et les renvois, `navigate` pointe l'état financier lui-même. |

`open_document` et `navigate` indiquent où lire sans ramener de passage ; le modèle enchaîne avec `read_page`. Contrairement à Mistral, aucun document n'est « ouvert » entre deux appels : chaque outil reçoit `doc`. Chaque passage ramené par `search` ou `read_page` reçoit un numéro `[n]`, affiché dans le résultat, que la réponse cite comme les autres. Les 10 passages initiaux gardent leurs numéros : les outils ne peuvent qu'**ajouter** après eux. Le rapport de vérification renvoyé avec chaque réponse liste les appels effectués.

### Le système inclut un retriever hybride pour maximiser la pertinence
- **Algo BM25 + Embeddings, côté Qdrant** : chaque chunk porte un vecteur sparse BM25 (précision lexicale) et un vecteur dense (sens contextuel) ; les deux recherches sont fusionnées par RRF dans une seule requête Qdrant. Métrique explicite (`VECTOR_SPACE = "cosine"`).
- **Un espace par utilisateur** : une collection Qdrant partagée, chaque point porte un `tenant_id` (index `is_tenant`), HNSW désactivé globalement (`m=0`) et construit par tenant (`payload_m`). Toutes les lectures passent par `backend/vectorstore/qdrant_store.py`, qui impose le filtre `tenant_id`. Les documents sont persistés : un fichier déjà indexé n'est ni ré-OCRisé ni ré-embeddé.
- **Routage par document** : avant de chercher, le système cible le(s) document(s) que la question désigne (nom d'entreprise ou de fichier) — indispensable quand plusieurs documents longs sont indexés ensemble.
- **Reranking Cohere + parent-child + multi-query** : petits chunks pour matcher, gros chunks pour répondre.

## Stack de modèles
Claude et les embeddings passent, au choix (`MODEL_PROVIDER`), par **Amazon Bedrock** (région
`eu-west-3`, par défaut) ou directement par les **API Anthropic et Cohere**. Les mêmes modèles
dans les deux cas :
- 💎 Claude Sonnet 4.6 (recherche à outils + génération) + Claude Haiku 4.5 (sous-agents : pertinence, routage, multi-query)
- ⚖️ Évaluation, toujours sur Bedrock : Pixtral Large 25.02 (juge LLM) et Ministral 3 14B (answer relevancy, en eu-west-1). Des Mistral plutôt que Claude : le juge ne note pas sa propre famille
- 🧠 Cohere Embed v4 (embeddings, 1024 dimensions) — identique des deux côtés : changer de fournisseur ne demande pas de réindexer

Toujours par leur propre API :
- ⚡ Mistral OCR (absent de Bedrock)
- 🧠 Cohere Rerank 4 Pro
- 🗄️ Qdrant Cloud (base vectorielle, un espace et un HNSW par utilisateur)

## Installation

1. **Cloner le projet** :
```bash
git clone https://github.com/julienlucas/agentic-rag-with-tools
```

2. **Installer les dépendances** :
```bash
uv sync
```

3. **Configuration** :

Choisissez d'abord le fournisseur des modèles Claude et des embeddings :

| `MODEL_PROVIDER` | Claude (Sonnet 4.6, Haiku 4.5) | Embeddings (Cohere Embed v4) | Prérequis |
|---|---|---|---|
| `bedrock` (défaut) | Amazon Bedrock | Amazon Bedrock | Compte AWS avec `bedrock:InvokeModel`, accès activé dans la console Bedrock à Claude Sonnet 4.6, Claude Haiku 4.5 et Cohere Embed v4 |
| `direct` | API Anthropic | API Cohere | Clé [platform.claude.com](https://platform.claude.com) |

Dans tous les cas :
- **Mistral** : une clé sur [console.mistral.ai](https://console.mistral.ai), pour l'OCR.
- **Cohere** : une clé sur [dashboard.cohere.com](https://dashboard.cohere.com), pour le reranker (et les embeddings en mode `direct`).
- **Qdrant** : un cluster sur [cloud.qdrant.io](https://cloud.qdrant.io) (AWS, région proche d'eu-west-3). Sans `QDRANT_URL`, Qdrant tourne en mémoire (données perdues au redémarrage).

Fichier `.env` :
```bash
# --- Fournisseur des modèles : bedrock (défaut) ou direct ---
MODEL_PROVIDER=bedrock

# --- Toujours requis ---
MISTRALAI_API_KEY=votre_clé_api_mistral_ici      # OCR
COHERE_API_KEY=votre_clé_api_cohere_ici          # reranker (+ embeddings en mode direct)
QDRANT_URL=https://xxxx.eu-central-1-0.aws.cloud.qdrant.io:6333
QDRANT_API_KEY=votre_clé_qdrant_ici

# --- MODEL_PROVIDER=bedrock ---
# Facultatif si ~/.aws/credentials ou un rôle IAM suffit. Présentes ici, ces clés passent
# AVANT les variables AWS du shell (un AWS_SESSION_TOKEN expiré dans le terminal ne gêne plus).
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=eu-west-3

# --- MODEL_PROVIDER=direct ---
ANTHROPIC_API_KEY=votre_clé_api_anthropic_ici
```

Réglages facultatifs (valeurs par défaut) :
```bash
# Identifiants de modèles : vides = défaut du fournisseur choisi.
#   bedrock : eu.anthropic.claude-sonnet-4-6, eu.anthropic.claude-haiku-4-5-20251001-v1:0, eu.cohere.embed-v4:0
#   direct  : claude-sonnet-4-6, claude-haiku-4-5, embed-v4.0
REASONING_MODEL_ID=
MODEL_SMALL_ID=
MODEL_ID=
EMBEDDING_MODEL_ID=
REASONING_EFFORT=medium          # low | medium | high | max
RERANK_MODEL=rerank-v4.0-pro
BM25_LANGUAGE=english            # langue du stemmer BM25 (réindexer si on la change)
TENANT_MAX_DOCUMENTS=50          # quotas par espace utilisateur
TENANT_MAX_CHUNKS=20000
QDRANT_CHUNKS_COLLECTION=chunks  # collections créées automatiquement au premier démarrage
QDRANT_PAGES_COLLECTION=pages
```

Pour surveiller votre application avec LangSmith (si vous le souhaitez) :

1. **Créer un compte LangSmith** : Allez sur [smith.langchain.com](https://smith.langchain.com)

2. **Obtenir votre clé API** : Dans les paramètres de votre compte

3. **Ajouter vos variables d'environnement**
```bash
# Configuration LangSmith pour le monitoring
LANGSMITH_API_KEY=votre_cle_api_langsmith_ici
LANGSMITH_PROJECT=agentic-search
```

4. **Lancer l'application** :
```bash
uv run python manage.py runserver
```

## Votes des utilisateurs : de 👎 au test de non-régression

Sous chaque réponse, un 👍 / 👎 (le 👎 ouvre un commentaire facultatif). Actifs seulement avec
`LANGSMITH_API_KEY` : chaque question est tracée dans le projet `agentic-search` sous un run racine
`question`, et le vote devient un feedback `user_score` (1 / 0) sur cette trace. Le navigateur ne voit
qu'un jeton signé qui lie la trace à son espace : en production, définir `DJANGO_SECRET_KEY`.

1. **Récolter** : `uv run python evaluation/feedback/collect.py` range les votes dans le dataset
   LangSmith « Retours utilisateurs », une fois par trace. Un 👍 y entre avec sa réponse comme
   référence. Un 👎 y entre avec sa réponse rejetée et le commentaire en métadonnées.
2. **Analyser** : chaque 👎 reçoit une cause d'échec proposée par le juge (Mistral, pas Claude) :
   `mauvais_document`, `passage_manquant`, `erreur_lecture`, `erreur_calcul`, `refus_a_tort`,
   `hors_corpus`, `forme` ou `autre`. Elle est aussi posée en feedback `failure_category` sur la trace,
   et `collect.py` affiche la répartition. La cause dit où corriger : routage, retrieval, prompt ou outils.
3. **Corriger** : `uv run python evaluation/feedback/replay.py` rejoue les exemples qui ont une
   référence, sur les mêmes documents du même espace, et les juge. Un 👍 qui n'est plus retrouvé est
   une **régression** (code de sortie 1). Un 👎 dont on a écrit la bonne réponse dans `outputs.answer`
   (depuis LangSmith) passe en `CORRIGÉ` le jour où le système la trouve. `--push` enregistre le rejeu
   comme expérience du dataset, comparable aux précédentes.

Les réponses 👍 ne sont pas injectées dans le prompt comme exemples : elles viennent des documents
d'un utilisateur et fuiraient vers les autres espaces. Elles servent de références de test.

## Évaluation FinanceBench (documents financiers difficiles)

L'évaluation, sur [FinanceBench](https://github.com/patronus-ai/financebench) (Patronus AI) —
le benchmark utilisé par [Mistral pour évaluer Agentic Search](https://mistral.ai/news/agentic-search/) :
QA sur des filings SEC de 150 à 260 pages, denses en tableaux.

**Préparation, l'indexation des documents (une seule fois, ~10-20 min)** — télécharge, OCRise et met en cache 4 10-K (AMD, American Express, Boeing, PepsiCo) :
```bash
uv run python evaluation/financebench/prepare.py
```

**Lancer l'évaluation (5-10 min)** :
```bash
uv run python evaluation/financebench/run_financebench_eval.py --mode both
```

**Résultats** — run du 28 septembre 2026, dans `evaluation/financebench/outputs_sonnet5/`.
Modèle de raisonnement : Claude Sonnet 5. 26 questions, 4 filings, index combiné, juge LLM
(Mistral Large) au protocole du benchmark, comptages bruts.
Les deux premières lignes sont l'ablation : même index, même retrieval, mêmes 10 passages
initiaux, la seule différence est le modèle de réponse avec ou sans outils.

| | Correctes | Hallucinations | Refus |
|---|---|---|---|
| Ce RAG, **avec** les outils (search / grep / read_page) | **96,2 % (25/26)** | 3,8 % (1/26) | 0 |
| Ce RAG, **sans** les outils (même retrieval, une seule génération) | 76,9 % (20/26) | 15,4 % (4/26) | 2 |
| Mistral Agentic Search — repère externe (Medium 3.5, 150 questions) | 86 % | | |
| Outils RAG juridiques commerciaux (étude Stanford) | 42-65 % | 17-33 % | |
| RAG naïf — papier FinanceBench (GPT-4-Turbo 2023, benchmark complet) | ~19 % | 81 % de réponses fausses ou refusées | |

Aucune erreur technique sur ce run : les 26 questions sont jugées dans les deux modes. Le run
précédent, sur Mistral Large (4 septembre 2026, `evaluation/financebench/outputs/`), donnait
83,3 % (20/24) avec outils et 65,4 % sans. Le retrieval mesuré diffère aussi entre les deux runs
(recall@5 : 20 % → 42 %) : l'écart ne mesure donc pas le seul effet du modèle.

Les lignes Mistral, Stanford et papier FinanceBench portent sur des échantillons différents de ce
RAG : ce sont des repères d'ordre de grandeur, pas un match à armes égales. Avec 26 questions,
l'intervalle de confiance à 95 % fait une trentaine de points : la comparaison qui tient est celle
des deux premières lignes, pas l'écart avec Mistral.

**Coût.** Le runner compte les tokens facturés de chaque appel (génération, sous-agents, juge LLM)
et les unités de recherche Cohere, et écrit le total dans `financebench_summary.json` (`cost`).
Mesuré le 28 septembre 2026 sur un run complet (26 questions, les deux modes, juge LLM compris) :

| Poste | Volume | Coût |
|---|---|---|
| Claude Sonnet 5 (réponse avec et sans outils) | 371 k tokens en entrée, 31 k en sortie | 1,05 $ |
| Mistral Large (juge) | 144 k tokens en entrée, 5 k en sortie | 0,08 $ |
| Mistral Small (sous-agents) | 61 k tokens | 0,01 $ |
| Cohere Rerank 4 Pro | 30 recherches à 0,0025 $ | 0,07 $ |
| **Un run complet** (10 minutes, 1 worker) | | **≈ 1,21 $ ≈ 1,04 €** |
| Préparation, une seule fois (OCR de 1 074 pages, embedding de 12 400 chunks) | | 4,4 $ ≈ 3,8 € |

Grilles publiques Anthropic (28 septembre 2026), La Plateforme et Cohere (5 septembre 2026), 1 $ = 0,86 €. Autrement dit :
l'évaluation complète, reproductible, sur quatre 10-K, se relance pour le prix d'un café, et le
chiffre de précision qu'on annonce à un client est re-mesurable à chaque changement de prompt.


Détail du protocole, options et notes d'implémentation : [`evaluation/financebench/README.md`](evaluation/financebench/README.md).
