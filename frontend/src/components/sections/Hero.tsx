import type { ReactNode } from "react";
import {
  ArrowDownUp,
  Redo2,
  ScanText,
  Search,
  ShieldCheck,
  Sparkles,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Container, Eyebrow } from "@/components/site/primitives";
import { DemoChat } from "@/components/rag/DemoChat";

type Vendor = "mistral" | "cohere";

/*
 * Le flow n'est plus un arc-en-ciel d'icônes : les six étapes partagent le même
 * cercle filet/encre, et seule la dernière — la réponse sourcée, ce que la page
 * vend — porte l'aplat d'or. La progression se lit au numéro, pas à la teinte.
 */
const steps: {
  icon: ReactNode;
  label: string;
  model: string;
  vendor: Vendor;
  accent?: boolean;
}[] = [
  { icon: <ScanText />, label: "OCR", model: "Mistral OCR", vendor: "mistral" },
  { icon: <Search />, label: "Recherche hybride", model: "Mistral Embed", vendor: "mistral" },
  { icon: <ArrowDownUp />, label: "Reranking", model: "Cohere Rerank v4 Pro", vendor: "cohere" },
  { icon: <ShieldCheck />, label: "Vérification de pertinence", model: "Mistral Small", vendor: "mistral" },
  { icon: <Redo2 />, label: "Recherche à outils", model: "Mistral Large", vendor: "mistral" },
  { icon: <Sparkles />, label: "Réponse sourcée", model: "Mistral Large", vendor: "mistral", accent: true },
];

const tooling = [
  "Reactjs",
  "LangGraph",
  "LangChain",
  "Chroma",
  "RAG BM25",
  "Django",
  "Modèles Mistral et Cohere",
];

const navLinks = [
  { href: "#demo", label: "Démo" },
  { href: "#agents", label: "Agents" },
  { href: "#resultats", label: "Résultats" },
  { href: "https://github.com/julienlucas/agentic-rag-with-tools", label: "Repo GitHub", external: true },
];

export function Hero() {
  return (
    <section id="top" className="border-b border-hairline bg-paper">
      <Container className="py-10 sm:py-16">
        <div className="flex items-start justify-between gap-10">
          <div className="min-w-0">
            <Eyebrow>Étude de cas · Julien Lucas</Eyebrow>
            {/* Le complément est une seconde ligne de titre : initiale minuscule et
                teinte adoucie, pour qu'il se lise comme la suite de la phrase et non
                comme un titre concurrent. */}
            <h1 className="display-xl mt-4">
              RAG <span className="accent-italic">agentique</span>
              <span className="display-md mt-2 block text-ink">
                Du RAG naïf au RAG agentique évalué stricte
              </span>
            </h1>
            <p className="copy mt-6">
              RAG agentique à outils inspiré de Mistral AI. Automatisation de
              réponses exigeant précision dans un corpus large de documents —
              documents techniques, rapports denses en tableaux, spécifications,
              appels d'offres.
            </p>
          </div>

          <nav
            aria-label="Sections"
            className="hidden shrink-0 flex-col items-end gap-3 pt-2 sm:flex"
          >
            {navLinks.map((l) => (
              <a
                key={l.href}
                href={l.href}
                target={l.external ? "_blank" : undefined}
                rel={l.external ? "noreferrer" : undefined}
                className="nav-link mono-xs uppercase"
              >
                {l.label}
                {l.external ? " ↗" : ""}
              </a>
            ))}
          </nav>
        </div>

        <h2 className="display-sm pb-4 mt-8">
          Un pipeline RAG agentique end-to-end
        </h2>
        <div
          id="demo"
          className="scroll-mt-20 rounded-[var(--radius-media)] border-none border-hairline"
        >
          <DemoChat>
            {/* Flow horizontal : six étapes reliées par un filet continu. */}
            <div className="-mx-5 mb-2 mt-4 overflow-x-auto px-5 sm:-mx-8 sm:px-8">
              <ol className="flex min-w-max items-stretch py-4 lg:min-w-0">
                {steps.map((s, i) => (
                  <li key={s.label} className="flex flex-1 items-start">
                    <div className="flex w-[9.5rem] flex-col items-center gap-3 px-2 text-center lg:w-auto lg:flex-1">
                      <span
                        className={cn(
                          "inline-flex size-11 items-center justify-center rounded-full transition-colors [&_svg]:size-[1.15rem]",
                          s.accent
                            ? "border-brand bg-brand text-on-ink"
                            : "border-hairline-strong bg-brand-surface text-brand-deep",
                        )}
                      >
                        {s.icon}
                      </span>
                      <span className="mono-xs text-ink-faint">
                        {String(i + 1).padStart(2, "0")}
                      </span>
                      <span className="text-[0.9275rem] font-medium leading-tight tracking-[-0.01em]">
                        {s.label}
                      </span>
                    </div>
                    {i < steps.length - 1 ? (
                      // Filet + pointe de flèche. Le conteneur fait 10 px de haut, décalé de
                      // 4,5 px vers le haut : le filet reste à mi-hauteur des cercles (44 px).
                      <span
                        aria-hidden
                        className="mt-[calc(1.375rem_-_4.5px)] flex h-2.5 min-w-4 flex-1 items-center text-hairline-strong"
                      >
                        <span className="h-px flex-1 bg-current" />
                        <svg
                          viewBox="0 0 6 10"
                          className="-ml-[5px] h-2.5 w-1.5 shrink-0"
                          fill="none"
                          stroke="currentColor"
                          strokeWidth="1.25"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        >
                          <path d="M1 1l4 4-4 4" />
                        </svg>
                      </span>
                    ) : null}
                  </li>
                ))}
              </ol>
            </div>
          </DemoChat>
        </div>

        <div className="mt-8 flex flex-wrap items-center gap-x-4 gap-y-3">
          <Eyebrow>Construit avec</Eyebrow>
          <ul className="flex flex-wrap items-center gap-1.5">
            {tooling.map((t) => (
              <li key={t} className="tag bg-brand-surface">
                {t}
              </li>
            ))}
          </ul>
        </div>
      </Container>
    </section>
  );
}
