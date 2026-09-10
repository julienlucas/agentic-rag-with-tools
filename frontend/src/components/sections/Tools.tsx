import { Section } from "@/components/site/primitives";

/*
 * Les outils, et pourquoi ils comptent. Chiffres Mistral : https://mistral.ai/news/agentic-search/
 * (20 août 2026), FinanceBench 150 questions, boucle complète (search + open / navigate / read /
 * grep) comparée à une boucle search seule. Chiffres de ce RAG : financebench_summary.json,
 * ablation à retrieval identique (baseline vs agentic).
 */
const tools = [
  {
    verb: "Trouver",
    signature: "search(query, doc)",
    mistral: "search",
    text: "Relance le retrieval hybride du pipeline (BM25 + vecteurs, routage, rerank Cohere) avec une requête reformulée, sur tout le corpus ou un seul document. 8 extraits, chacun avec son document et sa page.",
  },
  {
    verb: "Localiser",
    signature: "grep(pattern, doc)",
    mistral: "grep",
    text: "Toutes les occurrences littérales d'un terme, page par page, sur les 260 pages. Exhaustif : zéro résultat permet d'affirmer qu'une donnée est absente du rapport, au lieu de le supposer.",
  },
  {
    verb: "Lire",
    signature: "read_page(doc, page, end_page)",
    mistral: "open + navigate + read",
    text: "La page entière telle que l'OCR l'a produite, tableau compris, sur 1 à 3 pages quand il est à cheval. Ce qu'un chunk de 1 200 caractères ne montre jamais.",
  },
];

const toolGains = [
  {
    value: "−24 à −34 %",
    label: "de tokens consommés",
  },
  {
    value: "+7 à +9 pts",
    label: "de réponses correctes",
  },
  {
    value: "−40 %",
    label: "de latence",
    detail: "au p90 : 255 s → 154 s · en moyenne : 108 s → 71 s",
  },
];

const toolsHere = [
  {
    value: "65,4 → 83,3 %",
    label: "de réponses correctes",
    detail: "même index, même retrieval, mêmes 10 passages initiaux : la seule différence, ce sont les outils",
  },
  {
    value: "26,9 → 16,7 %",
    label: "d'hallucinations",
    detail: "7 réponses fausses sur 26 sans outils, 4 sur 24 avec",
  },
  {
    value: "9 sur 26",
    label: "questions ont appelé un outil",
    detail: "3,9 appels en moyenne, une page lue dans 8 cas sur 9 ; les 17 autres répondent en une passe, sans un token de plus",
  },
];

const limits = [
  {
    title: "Le recall du retrieval",
    text: "Sur 8 questions sur 26, la page de preuve n'atteint jamais le modèle, outils compris. Quand elle l'atteint, il répond juste dans 15 cas sur 18.",
  },
  {
    title: "Le modèle de raisonnement sur les questions est faible",
    text: (
      <>
        Un autre modèle de raisonnement que{" "}
        <img
          src="/static/mistral.png"
          alt="Mistral AI"
          className="inline-block h-4 w-auto align-text-bottom"
        />{" "}
        Ce qui est utilisé utilisé dans ce RAG agentique est Mistral Medium 3, un modèle de raisonnement limité. Pourtant un gros gain se joue ici.
      </>
    ),
  }
];

export function Tools() {
  return (
    <Section
      id="outils"
      index="03"
      eyebrow="Le levier déterminant"
      title={
        <>
          Ce RAG agentique ne se contente pas des passages qu&apos;on lui donne
          : il <span className="accent-italic">va chercher la preuve</span>,
          grâce à des outils.
        </>
      }
      intro="Un RAG classique choisit dix passages avant que le modèle ne lise la question, puis lui demande de répondre en une seule passe. Ici l'approche est différente, l'agent peut lui-même venir lire les documents, analyser un tableau, le contexte autour, partir, revenir, ect."
    >
      {/* d'où ça vient */}

      {/* les trois outils */}
      <div className="mt-4 grid gap-4 md:grid-cols-3">
        {tools.map((t, i) => (
          <div
            key={t.signature}
            className="card-paper border-hairline flex h-full flex-col p-6"
          >
            <div className="flex items-center justify-between">
              <span className="eyebrow">{t.verb}</span>
              <span className="mono-xs text-ink-faint">0{i + 1}</span>
            </div>
            <code className="mono-xs mt-3 w-fit rounded-sm bg-brand-surface px-2 py-1 text-brand-deep">
              {t.signature}
            </code>
            <p className="mb-4 mt-3 text-sm leading-relaxed text-ink-muted">
              {t.text}
            </p>
          </div>
        ))}
      </div>

      {/* moins de tokens, plus de précision */}
      <div className="card-paper border-hairline mt-10 p-8">
        <p className="display-sm mt-8">Voici les gains avec des outils</p>
        <p className="mono-xs mt-1 text-muted-foreground">
          Mistral · sur FinanceBench, 150 questions, 368 documents, soit 53900
          pages indéxées ensemble
        </p>
        <div className="mt-6 grid gap-6 sm:grid-cols-3">
          {toolGains.map((g) => (
            <div key={g.label} className="border-l-2 border-brand pl-4">
              <div className="font-display text-4xl font-normal tracking-tight tabular-nums">
                {g.value}
              </div>
              <div className="mt-1 text-sm font-medium">{g.label}</div>
              <div className="mono-xs mt-1 text-muted-foreground">
                {g.detail}
              </div>
            </div>
          ))}
        </div>
        <blockquote className="pt-6">
          <p className="display-sm text-ink">
            « Les outils de retrieval ne sont pas un surcoût : ils remplacent
            des recherches relancées pour rien par une{" "}
            <span className="accent-italic">navigation précise</span>. »
          </p>
          <cite className="mono-xs mt-2 block not-italic text-muted-foreground">
            Mistral AI, Introducing Agentic Search
          </cite>
        </blockquote>
      </div>

      <div className="mt-14">
        <div className="flex flex-col">
          <p className="display-md mt-3">Ce qui limite encore</p>
          <ol className="mt-6 divide-y divide-border border-t border-hairline">
            {limits.map((l, i) => (
              <li
                key={l.title}
                className="grid gap-1 py-4 sm:grid-cols-[2rem_16rem_1fr] sm:gap-4"
              >
                <span className="mono-xs pt-1 text-ink-faint">0{i + 1}</span>
                <span className="text-sm font-medium">{l.title}</span>
                <span className="text-sm leading-relaxed text-ink-muted">
                  {l.text}
                </span>
              </li>
            ))}
          </ol>
          <p className="mt-4 border-t border-sand pt-3 text-xs leading-relaxed text-muted-foreground">
            26 questions d'évaluation sur 4 filings : assez pour repérer les
            modes d'échec, trop peu pour se comparer à un benchmark complet.
          </p>
        </div>
      </div>
    </Section>
  );
}
