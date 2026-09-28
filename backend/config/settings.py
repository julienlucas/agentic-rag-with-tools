import os
from dotenv import load_dotenv
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()

class Settings(BaseSettings):
    # Paramètres requis
    MISTRALAI_API_KEY: str = os.getenv("MISTRALAI_API_KEY")
    MODEL_ID: str = "mistral-large-latest"  # HyDE, décomposition, compression, juge d'éval
    MODEL_SMALL_ID: str = "mistral-small-latest"  # Sous-agents (classif, reformulation)
    MODEL_OCR_ID: str = "mistral-ocr-latest"
    EMBEDDING_MODEL_ID: str = "mistral-embed"

    # Raisonnement (génération avec outils + agent de recherche) : Claude Sonnet 5.
    # Pas de `temperature` : Sonnet 5 la refuse (400). La réflexion adaptative est active
    # par défaut, ses tokens comptent dans max_tokens : d'où un plafond large, la longueur
    # de la réponse reste tenue par le prompt. Effort : low | medium | high | xhigh | max.
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    REASONING_MODEL_ID: str = "claude-sonnet-5"
    REASONING_EFFORT: str = "medium"
    REASONING_MAX_TOKENS: int = 8000
    REASONING_TIMEOUT: int = 90  # la réflexion allonge les appels, 30 s ne suffit pas

    # Timeouts et retries sur les appels LLM (évite les blocages de 2min)
    LLM_TIMEOUT: int = 30  # secondes par appel
    LLM_MAX_RETRIES: int = 2

    # Tracking LangSmith (si besoin)
    LANGSMITH_API_KEY: Optional[str] = None  # optionnel : le README le dit, le code ne le permettait pas

    # Paramètres optionnels avec valeurs par défaut

    CHROMA_COLLECTION_NAME: str = "documents"

    # Métrique de similarité de l'index vectoriel (Chroma -> HNSW : "cosine" | "l2" | "ip").
    # Posée explicitement à dessein : le défaut de Chroma est "l2", qui n'est correct que tant
    # que les embeddings sont normés. C'est le cas de mistral-embed (normes mesurées sur
    # l'index FinanceBench : 1 ± 2e-4), et sur des vecteurs unitaires L2 = 2 - 2·cos donne
    # exactement le même classement — vérifié sur 25 requêtes, top-20 identique.
    # Un modèle d'embedding non normé rendrait ce défaut faux SILENCIEUSEMENT : pas d'erreur,
    # pas de log, juste un recall qui baisse. D'où le réglage explicite.
    # ⚠️ Changer cette valeur invalide les index HNSW déjà persistés : il faut ré-embedder.
    VECTOR_SPACE: str = "cosine"

    # Cohere API (pour reranking)
    COHERE_API_KEY: Optional[str] = None

    # Paramètres d'embeddings

    # Paramètres de récupération - CONFIG OPTIMISÉE RECALL
    VECTOR_SEARCH_K: int = 20
    BM25_K: int = 20
    HYBRID_RETRIEVER_WEIGHTS: tuple = (0.5, 0.5)  # Équilibré — BM25 crucial pour termes exacts
    RERANK_ENABLED: bool = True
    RERANK_TOP_K: int = 30  # Top N résultats après reranking de TOUS les candidats
    RERANK_MODEL: str = "rerank-v4.0-pro"  # Cohere Rerank 4 Pro (gratuit en trial)

    # Multi-Query - 1 reformulation (compromis latence/recall pour la prod)
    MULTI_QUERY_ENABLED: bool = True
    MULTI_QUERY_COUNT: int = 1

    # HyDE - DÉSACTIVÉ (nuit au retrieval sur ce corpus)
    HYDE_ENABLED: bool = False

    # Query Decomposition - DÉSACTIVÉ
    QUERY_DECOMPOSITION_ENABLED: bool = False

    # Contextual Compression - DÉSACTIVÉ
    CONTEXTUAL_COMPRESSION_ENABLED: bool = False
    CONTEXTUAL_COMPRESSION_TOP_K: int = 5

    # Routage par document : avant de chercher, cibler le(s) document(s) que la question
    # désigne (nom d'entreprise / de fichier). Réduit la dilution quand plusieurs documents
    # longs sont indexés ensemble. Sans effet avec un seul document.
    DOCUMENT_ROUTING_ENABLED: bool = True

    # Recherche corrective (agentique) : si le vérificateur de pertinence juge les passages
    # insuffisants, réécrire la question dans le vocabulaire du document et relancer la
    # recherche, puis fusionner. Traite les questions dont les mots ne sont pas ceux du texte
    # ("legal battles" vs "litigation", "gross margin" absent d'un bilan bancaire).
    CORRECTIVE_RETRIEVAL_ENABLED: bool = True
    CORRECTIVE_MAX_ROUNDS: int = 1
    CORRECTIVE_QUERY_COUNT: int = 3
    # Les N premiers documents du retrieval initial sont intouchables : la recherche
    # corrective ne peut qu'ajouter après eux. Éval FinanceBench : la fusion RRF naïve
    # éjectait du top-10 des preuves initialement aux rangs 2 et 6.
    # = RESEARCH_TOP_K : la correction ne remplace JAMAIS un passage que le modèle aurait vu
    # sans elle. À 5, elle remplaçait les rangs 6-10 et perturbait la génération sur des
    # questions dont la preuve était déjà là (FinanceBench, 2 sept. 2026 : 2 CORRECT -> INCORRECT).
    CORRECTIVE_PROTECT_TOP: int = 10
    # Passages supplémentaires transmis au modèle après une correction, en plus des initiaux.
    # Avec l'agent à outils, une page entière lue par read_page compte pour un passage.
    CORRECTIVE_EXTRA_DOCS: int = 5
    # Le modèle de GÉNÉRATION reçoit lui-même les outils search / grep / read_page, sur
    # TOUTES les questions : il répond directement si le contexte suffit, sinon il cherche et
    # répond dans la même conversation (Agentic Search de Mistral). Remplace la recherche
    # corrective conditionnelle : sur le run du 4 sept. 2026, 3 des 6 questions sans preuve
    # étaient classées CAN_ANSWER par le vérificateur, l'agent n'y était jamais appelé.
    GENERATOR_TOOLS_ENABLED: bool = True
    GENERATOR_MAX_TOOL_CALLS: int = 5

    # Forme de la correction (quand GENERATOR_TOOLS_ENABLED est False) :
    #  - "agent"   : boucle d'outils (search / grep / read_page) menée par le modèle de
    #                génération, qui voit chaque résultat avant de décider du suivant.
    #  - "rewrite" : l'ancienne réécriture aveugle de la question par Mistral Small.
    # L'agent retombe sur "rewrite" si l'appel d'outils échoue (modèle sans function calling).
    CORRECTIVE_MODE: str = "agent"
    # Plafond d'appels d'outils par question corrigée.
    CORRECTIVE_MAX_TOOL_CALLS: int = 5
    # Déclencheur : score max du reranker Cohere sous ce seuil = le retrieval a
    # probablement raté -> corriger. (NO_MATCH du checker déclenche toujours ;
    # le checker seul ne suffit pas : son prompt le biaise vers PARTIAL et il ne
    # renvoie pratiquement jamais NO_MATCH.) 0 = désactive ce critère.
    # Filet pour un retrieval au score anormalement bas malgré un CAN_ANSWER.
    # À 0 = désactivé : PARTIAL et NO_MATCH suffisent à déclencher la correction, et ce
    # seuil-ci n'avait jamais rien déclenché sur les deux jeux d'éval.
    CORRECTIVE_RERANK_THRESHOLD: float = 0.0

    # Nombre de documents (parents) transmis au LLM pour la génération.
    # Éval FinanceBench : à 5, 3 questions sur 21 avaient leur preuve au rang 6-20.
    RESEARCH_TOP_K: int = 10

    # Paramètres de chunking - CONFIG OPTIMISÉE RECALL
    CHUNKING_STRATEGY: str = "semantic"
    PARENT_CHILD_ENABLED: bool = True
    PARENT_CHUNK_SIZE: int = 1200  # Parents ciblés pour le reranking
    CHILD_CHUNK_SIZE: int = 400  # Children assez gros pour matcher les keywords BM25
    CHILD_OVERLAP: int = 50
    CHUNK_SIZE: int = 500  # Réduit pour éviter la "moyennisation" des embeddings
    CHUNK_OVERLAP: int = 100  # 300 était excessif, 100 suffit
    SEMANTIC_THRESHOLD: float = 0.35  # Splits plus granulaires

    # Evaluation
    EVAL_LLM_JUDGE_ENABLED: bool = True


    # Nouveaux paramètres de cache avec annotations de type
    CACHE_DIR: str = "document_cache"
    CACHE_EXPIRE_DAYS: int = 7

    # Répertoire des exemples
    EXAMPLES_DIR: str = "./static"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()