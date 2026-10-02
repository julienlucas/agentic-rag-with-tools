import type { ReactNode } from "react";
import { ArrowDownUp, Redo2, ScanText, Search, ShieldCheck, Sparkles } from "lucide-react";
import { cn } from "@/lib/utils";

type Vendor = "mistral" | "cohere" | "anthropic";

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
  { icon: <Redo2 />, label: "Recherche à outils", model: "Claude Sonnet 5", vendor: "anthropic" },
  { icon: <Sparkles />, label: "Réponse sourcée", model: "Claude Sonnet 5", vendor: "anthropic", accent: true },
];

/** Schéma du pipeline : six étapes reliées par un filet continu. */
export function PipelineSteps({ className }: { className?: string }) {
  return (
    <div className={cn("-mx-5 overflow-x-auto px-5 sm:-mx-8 sm:px-8", className)}>
      <ol className="flex min-w-max items-stretch py-4 lg:min-w-0">
        {steps.map((s, i) => (
          <li key={s.label} className="flex flex-1 items-start">
            <div className="flex w-[9.5rem] flex-col items-center gap-3 px-2 text-center lg:w-auto lg:flex-1">
              <span
                className={cn(
                  "inline-flex size-11 items-center justify-center rounded-full transition-colors [&_svg]:size-[1.15rem]",
                  s.accent
                    ? "border-brand bg-brand text-on-brand"
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
    </div>  );
}
