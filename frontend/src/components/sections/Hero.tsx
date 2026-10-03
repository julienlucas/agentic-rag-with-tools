import { Container, Eyebrow } from "@/components/site/primitives";
import { DemoChat } from "@/components/rag/DemoChat";
import { Badge } from "@/components/ui/badge";

const tooling = [
  "Reactjs",
  "LangGraph",
  "LangChain",
  "Chroma",
  "BM25 + Vecteurs",
  "Django",
  "Amazon Bedrock",
  "Modèles Mistral et Cohere",
];

export function Hero() {
  return (
    <section id="top" className="border-b border-hairline bg-paper">
      <Container width="full" className="py-10 sm:py-16">
        {/* Centré, plus étroit que la nav. */}
        <div className="mx-auto max-w-5xl">
          <div className="min-w-0 text-center">
            {/* Le complément est une seconde ligne de titre : initiale minuscule et
                  teinte adoucie, pour qu'il se lise comme la suite de la phrase et non
                  comme un titre concurrent. */}
            <h1 className="display-xl">
              Agentic <span className="accent-italic">Search</span>
              <span className="display-xs mx-auto mt-4 mb-6 block max-w-3xl text-ink">
                <strong>Discutez avec vos documents</strong>, RAG agentique à ~96 % des réponses
                correctes, évalué sur un sous-ensemble de FinanceBench
              </span>
            </h1>
          </div>

          <div
            id="demo"
            className="scroll-mt-20 rounded-[var(--radius-media)] border-none border-hairline"
          >
            <DemoChat />
          </div>

          <div className="mt-8 flex flex-wrap items-center justify-center gap-x-4 gap-y-3">
            <Eyebrow>Construit avec</Eyebrow>
            <ul className="flex flex-wrap items-center justify-center gap-1.5">
              {tooling.map((t) => (
                <li key={t}>
                  <Badge variant="outlineBrand">{t}</Badge>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </Container>
    </section>
  );
}
