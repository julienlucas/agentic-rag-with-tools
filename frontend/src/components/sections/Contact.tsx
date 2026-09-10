import { ArrowUpRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Section } from "@/components/site/primitives";

const LINKEDIN = "https://www.linkedin.com/in/julien-lucas-jl";

export function Contact() {
  return (
    <Section
      id="contact"
      tone="ink"
      eyebrow="Contact"
      title="Qui je suis ?"
      intro={
        <>
          Julien Lucas. 5 ans comme développeur/software engineer en startups, scaleups, devenu AI Applied Engineer. Créateur de{" "}
          <a href="https://prospable.com" target="_blank" rel="noreferrer">
            Prospable
          </a>
          .
        </>
      }
    >
      <div className="flex flex-wrap items-center gap-6">
        <img
          src="/static/julienlucas.jpeg"
          alt="Julien Lucas"
          className="size-28 rounded-full border border-hairline object-cover shadow-card"
        />
        <div>
          <span className="display-sm block">Julien Lucas</span>
          <span className="mono-xs text-muted-foreground">
            AI Engineer
            <br />
            RAG, agents, automatisation et software engineer
          </span>
        </div>
        <Button asChild variant="brand" size="lg" className="ml-auto">
          <a href={LINKEDIN} target="_blank" rel="noreferrer">
            Me trouver sur LinkedIn <ArrowUpRight />
          </a>
        </Button>
      </div>
    </Section>
  );
}
