import { Redo2, ShieldCheck, Sparkles } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Section } from "@/components/site/primitives";

const agents = [
  {
    icon: <ShieldCheck />,
    index: "A1",
    title: "Agent vérificateur de pertinence",
    model: "mistral-small",
    text: "Lit les 3 meilleurs passages rerankés et décide s'ils répondent vraiment à la question : CAN_ANSWER, PARTIAL ou NO_MATCH. Son verdict est affiché dans le rapport et sert de signal ; il ne bloque la génération que s'il n'y a aucun passage.",
  },
  {
    icon: <Redo2 />,
    index: "A2",
    title: "Agent de recherche et de réponse",
    model: "mistral-large",
    text: "Reçoit les 10 meilleurs passages et trois outils — search (le retrieval hybride + rerank), grep (occurrences page par page, exhaustif) et read_page (la page entière, tableau compris, sur 1 à 3 pages). Il répond directement si le contexte suffit ; sinon il cherche, en voyant chaque résultat avant de décider du suivant (5 appels au plus), et répond dans la même conversation. Chaque passage ramené reçoit un numéro qu'il cite.",
  },
  {
    icon: <Sparkles />,
    index: "A3",
    title: "Génération contrainte",
    model: "mistral-large",
    text: "La réponse ne s'appuie que sur les passages numérotés, initiaux ou ramenés par les outils, avec une citation [n] après chaque affirmation. Elle refuse explicitement quand l'information n'y est pas — sauf pour calculer un ratio dont les composantes sont sous ses yeux, formule et chiffres cités.",
  },
];

export function Agents() {
  return (
    <Section
      id="agents"
      index="01"
      eyebrow="Les agents"
      title={
        <>
          Trois agents, orchestrés par <span className="accent-italic">LangGraph</span>.
        </>
      }
      intro="Le retriever remonte les passages ; le grand modèle décide s'il en a assez, cherche davantage s'il le faut, et rédige — dans la même conversation, comme dans l'Agentic Search de Mistral."
    >
      <div className="grid gap-4 md:grid-cols-3">
        {agents.map((a) => (
          <div key={a.title} className="card-paper border-hairline flex h-full flex-col p-6">
            <div className="flex items-center justify-between">
              <span className="inline-flex size-9 items-center justify-center rounded-md bg-brand-surface text-brand-deep [&_svg]:size-4.5">
                {a.icon}
              </span>
              <span className="mono-xs text-ink-faint">{a.index}</span>
            </div>
            <h3 className="display-sm mt-4">{a.title}</h3>
            <Badge variant="mono" className="mt-2 w-fit">
              {a.model}
            </Badge>
            <p className="mt-3 text-sm leading-relaxed text-ink-muted">{a.text}</p>
          </div>
        ))}
      </div>
    </Section>
  );
}
