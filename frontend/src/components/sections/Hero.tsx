import type { ReactNode } from "react";
import {
  ArrowDownUp,
  ArrowRight,
  Layers,
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

const steps: {
  icon: ReactNode;
  label: string;
  model: string;
  vendor: Vendor;
  /* teinte pleine très pâle (pas d'opacité) : bg-{couleur}-50 + text-{couleur}-500 */
  tone: string;
}[] = [
  {
    icon: <ScanText />,
    label: "OCR Mistral",
    model: "Mistral OCR",
    vendor: "mistral",
    tone: "bg-violet-50 text-violet-500",
  },
  {
    icon: <Search />,
    label: "Recherche hybride",
    model: "Mistral Embed",
    vendor: "mistral",
    tone: "bg-blue-50 text-blue-500",
  },
  {
    icon: <ArrowDownUp />,
    label: "Reranking Cohere",
    model: "Cohere Rerank v4 Pro",
    vendor: "cohere",
    tone: "bg-indigo-50 text-indigo-500",
  },
  {
    icon: <ShieldCheck />,
    label: "Vérification de pertinence",
    model: "Mistral Small",
    vendor: "mistral",
    tone: "bg-amber-50 text-amber-500",
  },
  {
    icon: <Redo2 />,
    label: "Recherche à outils",
    model: "Mistral Large",
    vendor: "mistral",
    tone: "bg-orange-50 text-orange-500",
  },
  {
    icon: <Sparkles />,
    label: "Réponse sourcée",
    model: "Mistral Large",
    vendor: "mistral",
    tone: "bg-emerald-50 text-emerald-500",
  },
];

function VendorLogo({ vendor }: { vendor: Vendor }) {
  if (vendor === "mistral") {
    return <img src="/static/mistral.png" alt="Mistral AI" className="size-4 rounded-sm object-contain" />;
  }
  return (
    <span className="grid size-4 place-items-center rounded-sm bg-ink text-[0.55rem] font-semibold leading-none text-on-ink">
      co
    </span>
  );
}

const tooling = [
  "Reactjs",
  "LangGraph",
  "LangChain",
  "Chroma",
  "RAG BM25",
  "Django",
  "Modèles Mistral et Cohere"
];

const navLinks = [
  { href: "#demo", label: "Démo" },
  { href: "#agents", label: "Agents" },
  { href: "#resultats", label: "Résultats" },
  { href: "https://github.com/julienlucas/agentic-rag-with-tools", label: "Repo GitHub", external: true },
];


export function Hero() {
  return (
    <section id="top" className="bg-brand-surface-strong/80 border-b border-border/30">
      <Container className="py-6 sm:py-14">
        <div className="mt-6 flex items-start justify-between gap-8">
          <div className="min-w-0">
            <h1 className="display-xl">
              RAG <span className="accent-italic">Agentique</span>
            </h1>
            <p className="copy mt-3 text-lg">
              Automatisations de réponses nécéssitant précision et citations sources
              (exemple: sur les documents techniques, rapports denses en
              tableaux, spécifications, appels d'offres, etc.)
            </p>
          </div>

          <nav
            aria-label="Sections"
            className="hidden shrink-0 flex-col items-end gap-1.5 pt-2 sm:flex"
          >
            {navLinks.map((l) => (
              <a
                key={l.href}
                href={l.href}
                target={l.external ? "_blank" : undefined}
                rel={l.external ? "noreferrer" : undefined}
                className="mono-xs text-muted-foreground transition-colors hover:text-brand-deep"
              >
                {l.label}
                {l.external ? " ↗" : ""}
              </a>
            ))}
          </nav>
        </div>

        <div
          id="demo"
          className="mt-6 scroll-mt-20 bg-brand-surface-strong/80 p-8 rounded-4xl"
        >
          <h2 className="display-md pb-3">
            Un pipeline RAG agentique end-to-end
          </h2>
          <DemoChat>
            {/* flow horizontal */}
            <div className="-mx-5 mb-4 mt-2 overflow-x-auto px-5 sm:-mx-8 sm:px-8">
              <ol className="flex min-w-max items-stretch lg:min-w-0 py-5 mt-3">
                {steps.map((s, i) => (
                  <li key={s.label} className="flex flex-1 items-center">
                    <div className="flex h-full w-[8.25rem] flex-col items-center gap-3 px-2 py-4 text-center lg:w-auto lg:flex-1">
                      <span
                        className={cn(
                          "inline-flex size-10 items-center justify-center rounded-full [&_svg]:size-5",
                          s.tone,
                        )}
                      >
                        {s.icon}
                      </span>
                      <span className="text-[1rem] !font-extrabold leading-tight">
                        {s.label}
                      </span>
                    </div>
                    {i < steps.length - 1 ? (
                      <ArrowRight
                        className="mx-1 -mt-9 size-4 shrink-0 text-sand-deep"
                        aria-hidden
                      />
                    ) : null}
                  </li>
                ))}
              </ol>
            </div>
          </DemoChat>
        </div>

        <div className="mt-4 flex flex-wrap items-center gap-x-4 gap-y-3">
          <span className="eyebrow">Construit avec</span>
          <ul className="flex flex-wrap items-center gap-1.5">
            {tooling.map((t) => (
              <li
                key={t}
                className="mono-xs rounded border border-border bg-paper px-2 py-1 text-ink-muted"
              >
                {t}
              </li>
            ))}
          </ul>
        </div>
      </Container>
    </section>
  );
}
