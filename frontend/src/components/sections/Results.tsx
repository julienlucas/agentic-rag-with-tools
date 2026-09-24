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
    label: "Ce RAG, sans outils",
    setup:
      "Même index, même retrieval, mêmes 10 passages initiaux, même modèle (Mistral Large) : une seule génération, sans search / grep / read_page. C'est le témoin qui mesure ce que les outils apportent.",
    correct: "65,4 %",
    wrong: "17 bonnes réponses sur 26 questions",
    tone: "plain" as const,
  },
  {
    label: "Ce RAG, avec outils",
    setup:
      "Modèle de raisonnement : Mistral Large (mistral-large-latest, La Plateforme), Mistral Small pour les sous-agents · OCR Mistral · chunking parent / enfant · hybride BM25 + vecteurs · routage · reranking Cohere · vérificateur de pertinence · agent de recherche à outils (search / grep / read_page) · génération contrainte aux preuves",
    correct: "83,3 %",
    wrong: "20 bonnes réponses sur 24 questions jugées",
    tone: "brand" as const,
  },
  {
    label: "Agentic Search · Mistral",
    setup:
      "Modèle de raisonnement : Mistral Medium 3.5 · 26,7 % en RAG one-shot sans boucle agentique, 86 % avec · boucle agentique + navigation · 150 questions sur les 368 documents (53900 pages) du benchmark complet + un benchmark de 89000 pages (OfficeQA Pro) qui sont des documents scannés",
    correct: "86 %",
    wrong: "évalué sur 150 questions sur FinanceBench",
    tone: "ref" as const,
  },
];

type Row = {
  metric: string;
  before: number;
  noTools: number;
  after: number;
  mistral: number;
  hint: string;
  lowerIsBetter?: boolean;
};

const rows: Row[] = [
  {
    metric: "Correctes",
    before: 19,
    noTools: 65.4,
    after: 83.3,
    mistral: 86,
    hint: "accuracy · verdict CORRECT du juge LLM · 20 sur 24 jugées (2 erreurs techniques exclues)",
  },
  {
    metric: "Fausses ou refusées",
    before: 81,
    noTools: 34.6,
    after: 16.7,
    mistral: 14,
    hint: "avec outils : 16,7 % d'hallucinations + 0 % de refus · 34,6 % sans outils (26,9 % d'hallucinations + 7,7 % de refus) · plus bas = mieux",
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

/*
 * Métriques détaillées du même run (4 sept. 2026), calculées a posteriori sur les réponses et les
 * pages sauvegardées : evaluation/financebench/README.md, sections « Faithfulness et answer
 * relevancy » et « Métriques de retrieval ». Le retrieval initial est le même dans les deux modes.
 */
type MetricRow = { metric: string; baseline: string; agentic: string; hint: string };

const metricGroups: { title: string; rows: MetricRow[] }[] = [
  {
    title: "Retrieval (10 passages initiaux, avant les outils)",
    rows: [
      {
        metric: "recall@5 · @10 · @20",
        baseline: "20,3 · 34,0 · 55,7 %",
        agentic: "identique",
        hint: "part des passages de preuve retrouvés dans les k premiers",
      },
      {
        metric: "precision@5 · @10 · @20",
        baseline: "14,6 · 13,1 · 8,7 %",
        agentic: "identique",
        hint: "part des k premiers passages issus d'une page de preuve · plafonnée : 1 à 2 pages de preuve par question",
      },
    ],
  },
  {
    title: "Réponse",
    rows: [
      {
        metric: "Correctes",
        baseline: "65,4 % (17)",
        agentic: "83,3 % (20)",
        hint: "verdict CORRECT du juge LLM · avec outils : 20 sur 24 jugées (2 erreurs techniques exclues)",
      },
      {
        metric: "Hallucinations",
        baseline: "26,9 % (7)",
        agentic: "16,7 % (4)",
        hint: "réponse affirmée mais fausse",
      },
      {
        metric: "Refus",
        baseline: "7,7 % (2)",
        agentic: "aucun",
        hint: "",
      },
      {
        metric: "Faithfulness",
        baseline: "4,73 / 5",
        agentic: "4,79 / 5",
        hint: "la réponse s'en tient aux extraits · note du juge LLM",
      },
      {
        metric: "Answer relevancy",
        baseline: "0,71",
        agentic: "0,81",
        hint: "la réponse traite la question posée · méthode RAGAS, 0 à 1",
      },
      {
        metric: "… sur les réponses correctes",
        baseline: "0,84 (17)",
        agentic: "0,80 (20)",
        hint: "",
      },
      {
        metric: "… sur les réponses fausses",
        baseline: "0,60 (7)",
        agentic: "0,84 (4)",
        hint: "les erreurs de l'agent restent centrées sur la question : mauvais chiffre, pas hors sujet",
      },
      {
        metric: "… sur les refus",
        baseline: "0 (2)",
        agentic: "aucun refus",
        hint: "un refus vaut 0 : l'essentiel de l'écart entre les deux modes",
      },
    ],
  },
];

function MetricsTable() {
  return (
    <figure className="card-paper border-hairline mt-14 p-8">
      <figcaption className="display-sm">Le détail des métriques</figcaption>
      <p className="mt-2 max-w-2xl text-xs leading-relaxed text-muted-foreground">
        Même run, mêmes 26 questions. « Sans outils » : une seule génération sur les 10 passages
        initiaux. « Avec outils » : le modèle cherche avec search / grep / read_page avant de
        répondre. Entre parenthèses, le nombre de réponses concernées.
      </p>
      <div className="mt-4 overflow-x-auto">
        <table className="w-full min-w-[34rem] text-sm">
          <thead>
            <tr className="text-left text-xs text-muted-foreground">
              <th className="py-2 pr-4 font-medium">Métrique</th>
              <th className="py-2 pr-4 font-medium">Sans outils</th>
              <th className="py-2 font-medium">Avec outils</th>
            </tr>
          </thead>
          {metricGroups.map((g) => (
            <tbody key={g.title}>
              <tr>
                <th colSpan={3} className="eyebrow pb-1 pt-5 text-left font-normal">
                  {g.title}
                </th>
              </tr>
              {g.rows.map((r) => (
                <tr key={r.metric} className="border-t border-hairline align-top">
                  <td className="py-2 pr-4">
                    <span className="font-medium">{r.metric}</span>
                    {r.hint && (
                      <span className="mt-0.5 block text-[0.7rem] leading-snug text-muted-foreground">
                        {r.hint}
                      </span>
                    )}
                  </td>
                  <td className="whitespace-nowrap py-2 pr-4 tabular-nums">{r.baseline}</td>
                  <td className="whitespace-nowrap py-2 tabular-nums font-medium">{r.agentic}</td>
                </tr>
              ))}
            </tbody>
          ))}
        </table>
      </div>
    </figure>
  );
}

function Bar({ value, tone, label }: { value: number; tone: "before" | "notools" | "after" | "mistral"; label: string }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <div className="flex h-5 items-center gap-2">
          <span
            className={cn(
              "h-2.5 rounded-r-[4px] transition-[width] duration-700",
              tone === "before" && "bg-chart-before",
              tone === "notools" && "bg-chart-notools",
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
  const d = row.after - row.noTools;
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
    <figure className="card-paper border-hairline p-8">
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
              <th className="py-2 font-medium">Sans outils</th>
              <th className="py-2 font-medium">Avec outils</th>
              <th className="py-2 font-medium">Δ outils</th>
              <th className="py-2 font-medium">Mistral Agentic Search</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.metric} className="border-t border-hairline">
                <td className="py-2 font-medium">{r.metric}</td>
                <td className="py-2 tabular-nums">
                  {r.before.toLocaleString("fr-FR")} %
                </td>
                <td className="py-2 tabular-nums">
                  {r.noTools.toLocaleString("fr-FR")} %
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
          <div className="mt-4 flex flex-wrap items-center gap-x-4 gap-y-1">
            <span className="flex items-center gap-1.5 text-xs text-ink-muted">
              <span className="size-2.5 rounded-sm bg-chart-before" /> RAG naïf
            </span>
            <span className="flex items-center gap-1.5 text-xs text-ink-muted">
              <span className="size-2.5 rounded-sm bg-chart-notools" /> Ce RAG sans outils
            </span>
            <span className="flex items-center gap-1.5 text-xs text-ink-muted">
              <span className="size-2.5 rounded-sm bg-chart-after" /> Ce RAG avec outils
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
                <div className="relative space-y-0.5 border-l border-hairline pl-3">
                  <Bar
                    value={r.before}
                    tone="before"
                    label={`${r.metric} · RAG naïf`}
                  />
                  <Bar
                    value={r.noTools}
                    tone="notools"
                    label={`${r.metric} · ce RAG sans outils`}
                  />
                  <Bar
                    value={r.after}
                    tone="after"
                    label={`${r.metric} · ce RAG avec outils`}
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
          <span className="accent-italic">FinanceBench</span> (limité à 4
          documents et 26 questions).
        </>
      }
      intro="FinanceBench est le benchmark que Mistral utilise pour évaluer leur outil Agentic Search : des questions financières sur des filings SEC denses en tableaux, où chaque chiffre apparaît des dizaines de fois. L'évaluation de mon RAG agentique porte ici sur 26 questions et 4 rapports (AMD, American Express, Boeing, PepsiCo, des documents de 150 à 260 pages)."
    >
      {/* trois niveaux */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {levels.map((l, i) => (
          <div
            key={l.label}
            className={cn(
              "flex h-full flex-col rounded-xl border p-8",
              l.tone === "muted" && "border-hairline bg-chart-after/5",
              l.tone === "plain" && "border-hairline bg-paper-2",
              l.tone === "brand" && "bg-chart-after/20 shadow-xl",
              l.tone === "ref" && "border-hairline bg-chart-ref/5",
            )}
          >
            <div className="flex items-center justify-between">
              <span className="eyebrow">{l.label}</span>
              <span className="mono-xs text-ink-faint">0{i + 1}</span>
            </div>
            <div className="font-display mt-4 text-5xl font-normal tracking-tight whitespace-nowrap">
              {l.correct}
            </div>
            <div className="mt-1 text-sm font-medium">réponses correctes</div>
            <div className="mono-xs mt-1 text-muted-foreground">{l.wrong}</div>
            <p className="mt-4 border-t border-hairline pt-3 text-xs leading-relaxed text-ink-muted">
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
      <figure className="pt-18 grid max-w-4xl gap-x-6 sm:grid-cols-[3.5rem_1fr]">
        <span
          aria-hidden
          className="display-xl -mt-3 hidden select-none leading-none text-brand sm:block"
        >
          &ldquo;
        </span>
        <div>
          <span className="eyebrow">Ce que les chiffres disent</span>
          <blockquote className="display-md mt-3 text-ink">
            Sur un sous-ensemble de FinanceBench (4 documents,{" "}
            <strong className="font-semibold text-ink">26 questions</strong>,
            index combiné), le système répond correctement à{" "}
            <span className="accent-italic">20 des 24 questions jugées</span>,
            avec 4 réponses fausses, aucun refus et 2 erreurs techniques
            exclues. Le RAG naïf plafonne est{" "}
            <span className="accent-italic">
              à ~19 % sur le benchmark complet
            </span>
            . Sans outils, le même système n'en répond que 17 sur 26 : les outils font gagner 18 points. Mistral Agentic Search (Mistral Medium 3.5)
            annonce 86 % sur{" "}
            <strong className="font-semibold text-ink">150 questions</strong> et
            368 documents — soit un périmètre bien plus large, qui n&apos;est
            pas comparable directement.
          </blockquote>
        </div>
      </figure>

      <MetricsTable />

      <div className="mt-16">
        <h3 className="display-md max-w-3xl">Ce qui a été fait</h3>
        <ol className="mt-6 space-y-1">
          {levers.map((l, i) => {
            // Du plus gros gain au plus faible : un or déjà léger au premier rang, qui
            // s'estompe jusqu'au papier au dernier. Le premier rang se marque au filet, pas
            // à la saturation — la teinte reste discrète sur toute la série.
            const strength = 0.2 - (i / (levers.length - 1)) * 0.185;
            const strong = i === 0;
            return (
              <li
                key={l.title}
                data-strong={strong || undefined}
                className="lever-row grid gap-1 rounded-sm px-4 py-4 sm:grid-cols-[2rem_16rem_1fr] sm:gap-4"
                style={
                  {
                    "--lever-strength": `${Math.round(strength * 100)}%`,
                  } as CSSProperties
                }
              >
                <span className="lever-index mono-xs pt-1">0{i + 1}</span>
                <span className="text-sm font-medium">{l.title}</span>
                <span className="lever-text text-sm leading-relaxed">
                  {l.text}
                </span>
              </li>
            );
          })}
        </ol>
      </div>
    </Section>
  );
}
