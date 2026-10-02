import { ArrowUpRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Container } from "@/components/site/primitives";
import { LINKEDIN } from "@/components/site/links";

export function AboutPage() {
  return (
    <>
      <section id="top" className="border-b border-hairline bg-paper">
        <Container width="full" className="py-14 sm:py-20">
          {/* Aligné à gauche sur la nav, mais plus étroit qu'elle. */}
          <div className="max-w-5xl">
            <span className="meta">à propos</span>
            <h1 className="display-xl mt-4">
              Agentic <span className="accent-italic">Search</span>
            </h1>
            <p className="copy mt-6 max-w-2xl">
              Un RAG agentique pour discuter avec de longs documents denses — rapports financiers,
              spécifications, appels d'offres — avec des réponses justes et sourcées à la page près.
              Le projet est open source, et chaque choix technique est mesuré sur un benchmark public.
            </p>
          </div>
        </Container>
      </section>

      <section id="contact" className="scroll-mt-16 bg-paper">
        <Container width="full" className="py-14 sm:py-20">
          <div className="max-w-5xl">
            <span className="meta">l'auteur</span>
            <div className="mt-6 flex flex-col gap-8 sm:flex-row sm:items-center">
              <img
                src="/static/julienlucas.jpeg"
                alt="Julien Lucas"
                className="size-32 shrink-0 rounded-full border border-hairline object-cover"
              />
              <div>
                <h2 className="display-lg font-semibold">Julien Lucas</h2>
                <p className="meta mt-3">
                  AI Engineer — RAG, agents, automatisation et software engineer
                </p>
              </div>
            </div>
            <p className="copy mt-8 max-w-2xl">
              5 ans comme développeur/software engineer en startups, scaleups, devenu AI Applied
              Engineer : RAG, agents, automatisation.
            </p>
            <Button asChild variant="brand" size="lg" className="mt-8">
              <a href={LINKEDIN} target="_blank" rel="noreferrer">
                Me trouver sur LinkedIn <ArrowUpRight />
              </a>
            </Button>
          </div>
        </Container>
      </section>
    </>
  );
}
