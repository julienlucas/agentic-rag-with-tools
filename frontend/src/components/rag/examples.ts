import type { Citation } from "@/api";

export type Evidence = { src: string; caption: string };

export type ExampleQuestion = {
  text: string;
  hint: string;
  evidence?: Evidence[];
};

export type ExampleDoc = {
  id: string;
  title: string;
  fileName: string;
  kind: "PDF";
  pages: number;
  type: string;
  description: string;
  questions: ExampleQuestion[];
};

export const exampleDocs: ExampleDoc[] = [
  {
    id: "deepseek-r1",
    title: "DeepSeek-R1 · rapport technique",
    fileName: "DeepSeek Technical Report.pdf",
    kind: "PDF",
    pages: 22,
    type: "Rapport technique",
    description:
      "Papier de recherche dense : tableaux de benchmarks, comparaisons multi-modèles, vocabulaire technique.",
    questions: [
      {
        text: "Résume l'évaluation des performances du modèle DeepSeek-R1 sur toutes les tâches de codage par rapport au modèle OpenAI o1-mini",
        hint: "Lecture croisée d'un tableau de benchmarks",
        evidence: [
          { src: "/static/deepseek-eval.png", caption: "Tableau d'évaluation, DeepSeek-R1 Technical Report" },
        ],
      },
      {
        text: "Quelle est la taille (nombre de paramètres) des modèles distillés à partir de DeepSeek-R1 et sur quels modèles de base sont-ils construits ?",
        hint: "Extraction de valeurs numériques dispersées",
      },
      {
        text: "Quel est le prix d'une action Nvidia aujourd'hui ?",
        hint: "Hors du document → le système doit refuser",
      },
    ],
  },
  {
    id: "google-env-2024",
    title: "Google · rapport environnemental 2024",
    fileName: "google-2024-environmental-report.pdf",
    kind: "PDF",
    pages: 86,
    type: "Rapport environnemental",
    description:
      "86 pages, tableaux régionaux (PUE, CFE), nombreuses valeurs proches qui piègent les LLM généralistes.",
    questions: [
      {
        text: "Récupère les valeurs d'efficacité PUE du centre de données dans l'installation 2 de Singapour en 2019 et 2022. Récupére également la moyenne régionale CFE en Asie-Pacifique en 2023",
        hint: "Deux tableaux, deux métriques, plusieurs années",
        evidence: [
          { src: "/static/google-pue.png", caption: "Tableau PUE par data center (p. Data centers)" },
          { src: "/static/google-cfe.png", caption: "Tableau CFE par région" },
        ],
      },
      {
        text: "Quel est l'objectif de Google en matière de réduction des émissions d'ici 2030, et où en est l'entreprise en 2023 ?",
        hint: "Synthèse à partir de plusieurs sections",
      },
    ],
  },
];

export const SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".txt", ".md"];
export const MAX_FILE_MB = 50;

/**
 * Sortie obtenue sur le document DeepSeek pré-chargé — affichée comme exemple tant qu'aucune
 * question n'est posée. Les citations [n] reprennent le format renvoyé par l'API ; leurs extraits
 * sont copiés du PDF (Table 4 p. 13 et commentaire p. 14).
 */
export const exampleOutput = {
  docId: "deepseek-r1",
  question:
    "Résume l'évaluation des performances du modèle DeepSeek-R1 sur toutes les tâches de codage par rapport au modèle OpenAI o1-mini",
  elapsed: 10,
  answer: `Voici l'évaluation des performances du modèle **DeepSeek-R1** par rapport à **OpenAI o1-mini** sur les tâches de codage, **exclusivement** d'après le contexte fourni :

### 1. Tâches de codage algorithmique
- **LiveCodeBench (Pass@1-COT)** : DeepSeek-R1 **65.9%** vs OpenAI o1-mini **53.8%** → DeepSeek-R1 surpasse o1-mini de **12.1 points** [1].
- **Codeforces (Percentile)** : DeepSeek-R1 **96.3%** vs o1-mini **93.4%** → légèrement supérieur (+2.9 points) [1].
- **Codeforces (Rating)** : DeepSeek-R1 **2029** vs o1-mini **1820** → +**209 points** [1].

### 2. Tâches de codage orientées ingénierie
- **SWE Verified (Resolved)** : DeepSeek-R1 **49.2%** vs o1-mini **41.6%** → +7.6 points [2].
- **Aider-Polyglot (Accuracy)** : DeepSeek-R1 **53.3%** vs o1-mini **32.9%** → +20.4 points [2].

### 3. Comparaison avec OpenAI o1-1217 (mentionné dans le contexte)
- Le contexte indique que **OpenAI o1-1217** est globalement supérieur à DeepSeek-R1 sur les tâches d'ingénierie (ex. Aider-Polyglot), mais que les performances sont **comparables** sur SWE Verified [3].`,
  report: `**Pertinent:** Oui
**Pertinence des passages:** CAN_ANSWER — les passages récupérés permettent de répondre à la question
**Confiance retrieval (reranker):** 0.95 — élevée
**Recherche corrective:** non nécessaire
**Sources utilisées:** DeepSeek Technical Report — 10 passages transmis au modèle`,
  citations: [
    {
      n: 1,
      source: "DeepSeek Technical Report.pdf",
      page: 12,
      locator: "DeepSeek Technical Report, p. 13",
      excerpt:
        "Table 4 · Code — LiveCodeBench (Pass@1-COT) : o1-mini 53.8, DeepSeek-R1 65.9 · Codeforces (Percentile) : 93.4, 96.3 · Codeforces (Rating) : 1820, 2029",
    },
    {
      n: 2,
      source: "DeepSeek Technical Report.pdf",
      page: 12,
      locator: "DeepSeek Technical Report, p. 13",
      excerpt:
        "Table 4 · Code — SWE Verified (Resolved) : o1-mini 41.6, DeepSeek-R1 49.2 · Aider-Polyglot (Acc.) : o1-mini 32.9, DeepSeek-R1 53.3",
    },
    {
      n: 3,
      source: "DeepSeek Technical Report.pdf",
      page: 13,
      locator: "DeepSeek Technical Report, p. 14",
      excerpt:
        "On engineering-oriented coding tasks, OpenAI-o1-1217 outperforms DeepSeek-R1 on Aider but achieves comparable performance on SWE Verified.",
    },
  ] satisfies Citation[],
};
