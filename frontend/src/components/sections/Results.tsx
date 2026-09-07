import { useState, type CSSProperties } from "react";
import { Table2 } from "lucide-react";
import { Section } from "@/components/site/primitives";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

/*
 * Chiffres : evaluation/financebench/outputs/financebench_summary.json — run du 4 sept. 2026,
 * 26 questions, 4 rapports 10-K, index combiné, juge LLM au protocole du benchmark.
 * Comptages bruts affichés à côté des pourcentages : on est sur 26 questions.
 * Le niveau « RAG naïf » est le chiffre publié dans le papier FinanceBench (Islam et al., 2023)
 * pour un RAG naïf sur vector store partagé, sur le benchmark complet — il n'a pas été re-mesuré
 * sur ce sous-ensemble.
 */
const levels = [
  {
    label: "RAG naïf",
    setup:
      "Vector store partagé, sans agents · chiffre du papier FinanceBench, benchmark complet",
    correct: "~19 %",
    wrong: "81 % de réponses fausses ou refusées",
    tone: "muted" as const,
  },
  {
    label: "Ce RAG agentique",
    setup:
      "OCR Mistral · chunking parent / enfant · hybride BM25 + vecteurs · routage · reranking Cohere · vérificateur de pertinence · agent de recherche à outils (search / grep / read_page) · génération contrainte aux preuves",
    correct: "83,3 %",
    wrong: "20 bonnes réponses sur 24 questions jugées",
    tone: "brand" as const,
  },
  {
    label: "Agentic Search · Mistral",
    setup:
      "Mistral Medium 3.5, boucle agentique + navigation · 150 questions sur les 368 documents (53900 pages) du benchmark complet + un benchmark de 89000 pages (OfficeQA Pro) qui sont des documents scannés",
    correct: "86 %",
    wrong: "évalué sur 150 questions sur FinanceBench",
    tone: "ref" as const,
  },
];

type Row = {
  metric: string;
  before: number;
  after: number;
  mistral: number;
  hint: string;
  lowerIsBetter?: boolean;
};

const rows: Row[] = [
  {
    metric: "Correctes",
    before: 19,
    after: 83.3,
    mistral: 86,
    hint: "accuracy · verdict CORRECT du juge LLM · 20 sur 24 jugées (2 erreurs techniques exclues)",
  },
  {
    metric: "Fausses ou refusées",
    before: 81,
    after: 16.7,
    mistral: 14,
    hint: "16,7 % d'hallucinations + 0 % de refus · plus bas = mieux",
    lowerIsBetter: true,
  },
];

const levers = [
  {
    title: "Un agent de recherche avec des outils",
    text: "Le modèle de réponse cherche lui-même avec search (hybride + rerank), grep (occurrences page par page) et read_page (la page entière, tableau compris, 1 à 3 pages), 5 appels au plus, et répond dans la même conversation. Sur ce run : outils appelés sur 9 questions sur 26, une page lue dans 8 cas sur 9, six questions gagnées sur la baseline.",
  },
  {
    title: "Reranking Cohere",
    text: "40 candidats rescorés, 30 conservés : les distracteurs sortent du top. Le plus gros gain côté retrieval.",
  },
  {
    title: "Recherche hybride + routage",
    text: "BM25 et vecteurs fusionnés par RRF, un routeur qui cible le bon document avant de chercher.",
  },
  {
    title: "OCR Mistral",
    text: "Les tableaux d'un 10-K survivent à l'extraction, en markdown, avec le numéro de page conservé sur chaque chunk.",
  },
  {
    title: "Génération contrainte",
    text: "Mistral Large ne répond qu'à partir des passages retenus et refuse quand la preuve manque — sauf pour calculer un ratio dont les composantes sont sous ses yeux, formule et chiffres cités.",
  },
  {
    title: "Chunking parent / enfant",
    text: "Petits chunks (400 car.) pour matcher, parents (1 200 car.) transmis au modèle pour répondre avec le contexte.",
  },
  {
    title: "Agent vérificateur de pertinence",
    text: "Les passages sont classés CAN_ANSWER / PARTIAL / NO_MATCH avant génération. Le verdict est transmis au modèle de réponse comme indice : sur PARTIAL ou NO_MATCH, il sait qu'il doit chercher.",
  },
];

const limits = [
  {
    title: "Le recall du retrieval",
    text: "Sur 8 questions sur 26, la page de preuve n'atteint jamais le modèle, outils compris. Quand elle l'atteint, il répond juste dans 15 cas sur 18.",
  },
  {
    title: "Le modèle est faible",
    text: (
      <>
        Un autre modèle de raisonnement que{" "}
        <img
          src="/static/mistral.png"
          alt="Mistral AI"
          className="inline-block h-4 w-auto align-text-bottom"
        />{" "}
        Ce qui est utilisé utilisé dans ce RAG agentique est Mistral Medium 3, un modèle de raisonnement limité. Pourtant un gros gain se joue ici.
        Actuellement à contexte strictement identique, des réponses
        changent de verdict d&apos;un run à l&apos;autre. Un écart d&apos;une ou deux questions ne
        se lit donc pas comme une amélioration.
      </>
    ),
  },
  {
    title: "Le coût en latence",
    text: "Les outils rendent la génération 2,7× plus lente en moyenne : 12,7 s contre 4,6 s par question, le prix de 3 à 5 appels sur un tiers des questions. Les autres ne changent pas.",
  },
];

function Bar({ value, tone, label }: { value: number; tone: "before" | "after" | "mistral"; label: string }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <div className="flex h-5 items-center gap-2">
          <span
            className={cn(
              "h-2.5 rounded-r-[4px] transition-[width] duration-700",
              tone === "before" && "bg-chart-before",
              tone === "after" && "bg-chart-after",
              tone === "mistral" && "bg-chart-ref",
            )}
            style={{ width: `${value}%` }}
          />
          <span className="mono-xs shrink-0 whitespace-nowrap tabular-nums text-ink-muted">{value.toLocaleString("fr-FR")} %</span>
        </div>
      </TooltipTrigger>
      <TooltipContent>
        {label} · {value.toLocaleString("fr-FR")} %
      </TooltipContent>
    </Tooltip>
  );
}

function Delta({ row }: { row: Row }) {
  const d = row.after - row.before;
  const good = row.lowerIsBetter ? d < 0 : d > 0;
  return (
    <span className={cn("tabular-nums", d === 0 ? "text-muted-foreground" : good ? "text-brand-deep" : "text-destructive")}>
      {d === 0 ? "=" : `${d > 0 ? "+" : "−"}${Math.abs(d).toLocaleString("fr-FR", { maximumFractionDigits: 1 })} pts`}
    </span>
  );
}

function BenchmarkChart() {
  const [table, setTable] = useState(false);
  return (
    <figure className="card-paper border-border/30 p-8">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0 max-w-2xl">
          <figcaption className="display-sm">
            Du RAG naïf au système agentique
          </figcaption>
        </div>
        <Button
          variant="ghost"
          size="sm"
          className="ml-auto shrink-0 -mt-1"
          onClick={() => setTable((t) => !t)}
          aria-pressed={table}
        >
          <Table2 /> {table ? "Graphique" : "Tableau"}
        </Button>
      </div>

      {table ? (
        <table className="mt-4 w-full text-sm">
          <thead>
            <tr className="text-left text-xs text-muted-foreground">
              <th className="py-2 font-medium">Métrique</th>
              <th className="py-2 font-medium">RAG naïf</th>
              <th className="py-2 font-medium">RAG agentique</th>
              <th className="py-2 font-medium">Δ</th>
              <th className="py-2 font-medium">Mistral Agentic Search</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.metric} className="border-t border-border/30">
                <td className="py-2 font-medium">{r.metric}</td>
                <td className="py-2 tabular-nums">
                  {r.before.toLocaleString("fr-FR")} %
                </td>
                <td className="py-2 tabular-nums">
                  {r.after.toLocaleString("fr-FR")} %
                </td>
                <td className="py-2">
                  <Delta row={r} />
                </td>
                <td className="py-2 tabular-nums text-ink-muted">
                  {r.mistral.toLocaleString("fr-FR")} %
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <>
          <div className="mt-4 flex items-center gap-4">
            <span className="flex items-center gap-1.5 text-xs text-ink-muted">
              <span className="size-2.5 rounded-sm bg-chart-before" /> RAG naïf
            </span>
            <span className="flex items-center gap-1.5 text-xs text-ink-muted">
              <span className="size-2.5 rounded-sm bg-chart-after" /> RAG agentique
            </span>
            <span className="flex items-center gap-1.5 text-xs text-ink-muted">
              <span className="size-2.5 rounded-sm bg-chart-ref" /> Mistral
              Agentic Search
            </span>
          </div>
          <div className="mt-5 space-y-5">
            {rows.map((r) => (
              <div
                key={r.metric}
                className="grid gap-1 sm:grid-cols-[11rem_1fr]"
              >
                <div>
                  {/* nowrap : « Fausses ou refusées » passait à la ligne et désalignait
                      les deux libellés. La colonne est commune, donc ils restent alignés. */}
                  <span className="whitespace-nowrap text-sm font-medium">
                    {r.metric}
                  </span>
                  <span className="mt-0.5 block text-[0.7rem] leading-snug text-muted-foreground">
                    {r.hint}
                  </span>
                </div>
                <div className="relative space-y-0.5 border-l border-border pl-3">
                  <Bar
                    value={r.before}
                    tone="before"
                    label={`${r.metric} · avant`}
                  />
                  <Bar
                    value={r.after}
                    tone="after"
                    label={`${r.metric} · après`}
                  />
                  <Bar
                    value={r.mistral}
                    tone="mistral"
                    label={`${r.metric} · Mistral Agentic Search`}
                  />
                </div>
              </div>
            ))}
          </div>
        </>
      )}
    </figure>
  );
}

export function Results() {
  return (
    <Section
      id="resultats"
      index="02"
      eyebrow="L'évaluation"
      title={
        <>
          Évalué à 83,3 % de réponses correctes sur le benchmark{" "}
          <span className="accent-italic">FinanceBench</span> (sur 26 questions).
        </>
      }
      intro="FinanceBench est le benchmark que Mistral utilise pour évaluer leur outil Agentic Search : des questions financières sur des filings SEC denses en tableaux, où chaque chiffre apparaît des dizaines de fois. L'évaluation de mon RAG agentique porte ici sur 26 questions et 4 rapports (AMD, American Express, Boeing, PepsiCo, des documents de 150 à 260 pages)."
    >
      {/* trois niveaux */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {levels.map((l, i) => (
          <div
            key={l.label}
            className={cn(
              "flex h-full flex-col rounded-xl border p-8",
              l.tone === "muted" && "border-border/30 bg-paper-2",
              l.tone === "brand" && "border-brand/10 bg-paper shadow-card",
              l.tone === "ref" &&
                "border-dashed/30 border-chart-ref/20 bg-paper-2",
            )}
          >
            <div className="flex items-center justify-between">
              <span className="eyebrow">{l.label}</span>
              <span className="mono-xs text-sand-deep">0{i + 1}</span>
            </div>
            <div className="font-display mt-4 text-5xl font-normal tracking-tight">
              {l.correct}
            </div>
            <div className="mt-1 text-sm font-medium">réponses correctes</div>
            <div className="mono-xs mt-1 text-muted-foreground">{l.wrong}</div>
            <p className="mt-4 border-t border-border pt-3 text-xs leading-relaxed text-ink-muted">
              {l.setup}
            </p>
          </div>
        ))}
      </div>
      {/* RAG naïf vs système agentique */}
      <div className="mt-14">
        <BenchmarkChart />
      </div>

      {/* Ce que les chiffres autorisent à dire — et pas plus. Hors de la carte, exprès. */}
      <figure className="py-18 grid max-w-4xl gap-x-6 sm:grid-cols-[3.5rem_1fr]">
        <span
          aria-hidden
          className="display-xl -mt-3 hidden select-none leading-none text-brand sm:block"
        >
          &ldquo;
        </span>
        <div>
          <span className="eyebrow">Ce que les chiffres autorisent à dire</span>
          <blockquote className="display-md mt-3 text-ink">
            Sur un sous-ensemble de FinanceBench (4 filings,{" "}
            <strong className="font-semibold text-ink">26 questions</strong>,
            index combiné), le système répond correctement à{" "}
            <span className="accent-italic">20 des 24 questions jugées</span>,
            avec 4 réponses fausses, aucun refus et 2 erreurs techniques
            exclues. Le même système sans les outils, à retrieval identique, est
            à <span className="accent-italic">17 sur 26</span> avec 7 réponses
            fausses : c&apos;est l&apos;écart qui compte. Le RAG naïf du papier
            est à ~19 % sur le benchmark complet ; Mistral Agentic Search
            annonce 86 % sur{" "}
            <strong className="font-semibold text-ink">150 questions</strong> et
            368 filings — un périmètre bien plus large, qui n&apos;est pas
            comparable directement.
          </blockquote>
        </div>
      </figure>

      <div className="mt-14">
        <h3 className="display-md max-w-3xl">
          Ce qui a été fait pour passer à ce résultat :
        </h3>
        <ol className="mt-6 space-y-1">
          {levers.map((l, i) => {
            // Du plus foncé (plus gros gain) au plus clair : la teinte de marque s'estompe à chaque rang.
            const strength = 1 - (i / (levers.length - 1)) * 0.92;
            const strong = strength > 0.6;
            return (
              <li
                key={l.title}
                data-strong={strong || undefined}
                className="lever-row grid gap-1 rounded-md px-4 py-4 sm:grid-cols-[2rem_16rem_1fr] sm:gap-4"
                style={{ "--lever-strength": `${Math.round(strength * 100)}%` } as CSSProperties}
              >
                <span className="lever-index mono-xs pt-1">0{i + 1}</span>
                <span className="text-sm font-medium">{l.title}</span>
                <span className="lever-text text-sm leading-relaxed">{l.text}</span>
              </li>
            );
          })}
        </ol>
      </div>

      <div className="mt-14">
        <div className="flex flex-col">
          <p className="display-md mt-3">Ce qui limite encore</p>
          <ol className="mt-6 divide-y divide-border border-t border-border">
            {limits.map((l, i) => (
              <li
                key={l.title}
                className="grid gap-1 py-4 sm:grid-cols-[2rem_16rem_1fr] sm:gap-4"
              >
                <span className="mono-xs pt-1 text-sand-deep">0{i + 1}</span>
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
