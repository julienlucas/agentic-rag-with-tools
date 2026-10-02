import os
from dotenv import load_dotenv
from typing import Literal, Optional
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()

# Modèles par défaut de chaque fournisseur (mêmes modèles, identifiants différents).
PROVIDER_DEFAULTS = {
    "bedrock": {
        "MODEL_ID": "eu.anthropic.claude-haiku-4-5-20251001-v1:0",
        "MODEL_SMALL_ID": "eu.anthropic.claude-haiku-4-5-20251001-v1:0",
        "REASONING_MODEL_ID": "eu.anthropic.claude-sonnet-4-6",
        "EMBEDDING_MODEL_ID": "eu.cohere.embed-v4:0",
    },
    "direct": {
        "MODEL_ID": "claude-haiku-4-5",
        "MODEL_SMALL_ID": "claude-haiku-4-5",
        "REASONING_MODEL_ID": "claude-sonnet-4-6",
        "EMBEDDING_MODEL_ID": "embed-v4.0",
    },
}

class Settings(BaseSettings):
    # Fournisseur des modèles Claude et des embeddings :
    #  - "bedrock" : Amazon Bedrock (identifiants AWS du .env, sinon chaîne standard de boto3).
    #    Profils d'inférence « eu. » : en eu-west-3, l'appel direct à l'ID du modèle est refusé.
    #  - "direct"  : API Anthropic (ANTHROPIC_API_KEY) et API Cohere (COHERE_API_KEY).
    # Les embeddings sont le même modèle des deux côtés (Cohere Embed v4, 1024 dimensions) :
    # changer de fournisseur ne demande pas de réindexer Qdrant.
    MODEL_PROVIDER: Literal["bedrock", "direct"] = "bedrock"
    AWS_REGION: str = "eu-west-3"
    ANTHROPIC_API_KEY: Optional[str] = None

    # Identifiants de modèles : vides = défaut du fournisseur (PROVIDER_DEFAULTS ci-dessous).
    MODEL_ID: Optional[str] = None  # HyDE, décomposition, compression
    MODEL_SMALL_ID: Optional[str] = None  # Sous-agents (classif, reformulation)
    EMBEDDING_MODEL_ID: Optional[str] = None
    EMBEDDING_DIMENSIONS: int = 1024

    # Raisonnement (génération avec outils + agent de recherche) : Claude Sonnet 4.6.
    # Réflexion adaptative : ses tokens comptent dans max_tokens, d'où un plafond large, la
    # longueur de la réponse reste tenue par le prompt. Effort : low | medium | high | max
    # (pas de xhigh sur Sonnet 4.6).
    REASONING_MODEL_ID: Optional[str] = None
    REASONING_EFFORT: str = "medium"
    REASONING_MAX_TOKENS: int = 8000
    REASONING_TIMEOUT: int = 90  # la réflexion allonge les appels, 30 s ne suffit pas

    # OCR : Mistral OCR par son API (absent de Bedrock). Le juge de l'évaluation reste aussi
    # sur Mistral Large, pour que les scores FinanceBench restent comparables aux runs passés.
    MISTRALAI_API_KEY: Optional[str] = os.getenv("MISTRALAI_API_KEY")
    MODEL_OCR_ID: str = "mistral-ocr-latest"
    EVAL_JUDGE_MODEL_ID: str = "mistral-large-latest"

    # Timeouts et retries sur les appels LLM (évite les blocages de 2min)
    LLM_TIMEOUT: int = 30  # secondes par appel
    LLM_MAX_RETRIES: int = 2

    # Tracking LangSmith (si besoin)
    LANGSMITH_API_KEY: Optional[str] = None  # optionnel : le README le dit, le code ne le permettait pas

    # Paramètres optionnels avec valeurs par défaut

    # Qdrant Cloud : une collection partagée, isolée par tenant_id (multi-tenancy Qdrant).
    QDRANT_URL: Optional[str] = None  # None -> Qdrant en mémoire (tests, dev sans cluster)
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_CHUNKS_COLLECTION: str = "chunks"
    QDRANT_PAGES_COLLECTION: str = "pages"  # pages OCR (grep / read_page) + registre des documents
    # HNSW : pas de graphe global (m=0), un graphe par tenant (payload_m) — chaque utilisateur
    # a son index, et une recherche ne parcourt jamais les vecteurs des autres.
    QDRANT_HNSW_PAYLOAD_M: int = 16
    QDRANT_HNSW_EF_CONSTRUCT: int = 100
    # Quotas par tenant : le cluster gratuit fait ~1 Go de RAM pour tous les utilisateurs.
    TENANT_MAX_DOCUMENTS: int = 50
    TENANT_MAX_CHUNKS: int = 20000
    # Vecteur sparse BM25 (fastembed « Qdrant/bm25 ») : la langue fixe stemmer et stopwords.
    BM25_LANGUAGE: str = "english"

    # Métrique de similarité de l'index vectoriel. Cohere Embed v4 n'est pas garanti normé :
    # cosinus explicite. ⚠️ La changer impose de recréer la collection et de ré-embedder.
    VECTOR_SPACE: str = "cosine"

    # Rerank : Cohere Rerank 4 Pro, par l'API Cohere (pas Bedrock, qui ne propose que la 3.5).
    COHERE_API_KEY: Optional[str] = None

    # Paramètres d'embeddings

    # Paramètres de récupération - CONFIG OPTIMISÉE RECALL
    VECTOR_SEARCH_K: int = 20
    BM25_K: int = 20
    HYBRID_RETRIEVER_WEIGHTS: tuple = (0.5, 0.5)  # Équilibré — BM25 crucial pour termes exacts
    RERANK_ENABLED: bool = True
    RERANK_TOP_K: int = 30  # Top N résultats après reranking de TOUS les candidats
    RERANK_MODEL: str = "rerank-v4.0-pro"

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

    @model_validator(mode="after")
    def _provider_defaults(self):
        for field, value in PROVIDER_DEFAULTS[self.MODEL_PROVIDER].items():
            if not getattr(self, field):
                setattr(self, field, value)
        return self

settings = Settings()