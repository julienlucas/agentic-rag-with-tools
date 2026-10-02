import { useState } from "react";
import { Menu, X } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Container } from "@/components/site/primitives";
import { GITHUB_REPO } from "@/components/site/links";

const links = [
  { href: "/#resultats", label: "Résultats" },
  { href: GITHUB_REPO, label: "GitHub", external: true },
];

/*
 * Barre de navigation collante : fond papier légèrement translucide, filet en pied de barre.
 * Sous `sm`, les liens passent dans un panneau déroulant.
 */
export function Navbar() {
  const [open, setOpen] = useState(false);

  return (
    <header
      className={cn(
        "sticky top-0 z-40 border-b bg-paper/90 backdrop-blur supports-[backdrop-filter]:bg-paper/75",
        "transition-[border-color] duration-[var(--dur-fast)]",
        "border-hairline",
      )}
    >
      <Container width="full" className="flex h-14 items-center justify-between gap-6">
        <a href="/" className="text-[1.6275rem] font-semibold tracking-[-0.035em] text-ink" onClick={() => setOpen(false)}>
          Agentic <span className="accent-italic">Search</span>
        </a>

        <nav aria-label="Navigation principale" className="hidden items-center gap-7 sm:flex">
          {links.map((l) => (
            <a
              key={l.href}
              href={l.href}
              target={l.external ? "_blank" : undefined}
              rel={l.external ? "noreferrer" : undefined}
              className="nav-link font-sans text-sm font-medium"
            >
              {l.label}
              {l.external ? <span className="text-brand"> ↗</span> : null}
            </a>
          ))}
          <Button asChild size="sm" variant="outline">
            <a href="/a-propos#contact">Contact</a>
          </Button>
        </nav>

        <button
          type="button"
          className="inline-flex size-9 items-center justify-center rounded-sm text-ink sm:hidden"
          aria-label={open ? "Fermer le menu" : "Ouvrir le menu"}
          aria-expanded={open}
          onClick={() => setOpen((v) => !v)}
        >
          {open ? <X className="size-5" /> : <Menu className="size-5" />}
        </button>
      </Container>

      {open ? (
        <nav aria-label="Navigation principale" className="border-t border-hairline sm:hidden">
          <Container width="full" className="flex flex-col py-2">
            {links.map((l) => (
              <a
                key={l.href}
                href={l.href}
                target={l.external ? "_blank" : undefined}
                rel={l.external ? "noreferrer" : undefined}
                className="nav-link font-sans py-3 text-sm font-medium"
                onClick={() => setOpen(false)}
              >
                {l.label}
                {l.external ? <span className="text-brand"> ↗</span> : null}
              </a>
            ))}
            <Button asChild size="sm" variant="outline" className="my-2 self-start">
              <a href="/a-propos#contact" onClick={() => setOpen(false)}>Contact</a>
            </Button>
          </Container>
        </nav>
      ) : null}
    </header>
  );
}
