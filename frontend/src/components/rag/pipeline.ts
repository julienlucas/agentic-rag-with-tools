/**
 * Étapes du pipeline telles qu'implémentées côté backend
 * (retriever/builder.py + agents/workflow.py + config/settings.py).
 * Les durées servent uniquement à animer la trace pendant l'attente :
 * le backend ne streame pas, les signaux réels arrivent avec la réponse.
 */
export type PipelineStep = {
  key: "route" | "retrieve" | "rerank" | "check" | "correct" | "research";
  label: string;
  detail: string;
  model?: string;
  conditional?: boolean;
  ms: number;
};

export const pipelineSteps: PipelineStep[] = [
  {
    key: "route",
    label: "Routage & réécriture",
    detail:
      "Cible le document visé par la question, puis génère une reformulation (multi-query) pour élargir le rappel.",
    model: "mistral-small",
    ms: 1400,
  },
  {
    key: "retrieve",
    label: "Recherche hybride",
    detail:
      "BM25 (k = 20) + vecteurs Chroma / mistral-embed (k = 20), fusion RRF 0,5 / 0,5, puis remontée des chunks parents (1 200 car.).",
    model: "mistral-embed",
    ms: 1800,
  },
  {
    key: "rerank",
    label: "Reranking Cohere",
    detail: "rerank-v4.0-pro sur ≤ 40 candidats → 30 passages scorés.",
    model: "cohere rerank-v4",
    ms: 1600,
  },
  {
    key: "check",
    label: "Agent vérificateur de pertinence",
    detail: "Classe les 3 meilleurs passages : CAN_ANSWER, PARTIAL ou NO_MATCH — signal affiché dans le rapport.",
    model: "mistral-small",
    ms: 2200,
  },
  {
    key: "correct",
    label: "Recherche à outils",
    detail:
      "Si le contexte ne suffit pas, le modèle enchaîne search / grep / read_page / open_document / navigate (5 appels max) ; chaque passage ramené est numéroté et s'ajoute après les 10 initiaux.",
    model: "claude-sonnet-5",
    conditional: true,
    ms: 0,
  },
  {
    key: "research",
    label: "Agent de recherche et de réponse",
    detail:
      "Reçoit les 10 meilleurs passages et les outils, rédige la réponse avec citations [n] — refuse explicitement si la preuve est absente.",
    model: "claude-sonnet-5",
    ms: 0,
  },
];
