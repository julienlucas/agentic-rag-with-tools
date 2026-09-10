import { useState } from "react";
import { Calendar, Mail, X } from "lucide-react";
import { Button } from "@/components/ui/button";

export function ContactWidget() {
  const [open, setOpen] = useState(false);

  const goToContact = () => {
    document.getElementById("contact")?.scrollIntoView({ behavior: "smooth", block: "start" });
    setOpen(false);
  };

  return (
    <div className="fixed bottom-5 right-5 z-40 flex flex-col items-end gap-3">
      {open ? (
        <div className="card-paper rise-in relative w-[268px] p-4 text-sm shadow-lift">
          <button
            type="button"
            onClick={() => setOpen(false)}
            className="absolute right-2.5 top-2.5 rounded-md p-1 text-muted-foreground transition-colors hover:bg-paper-2 hover:text-foreground"
            aria-label="Fermer"
          >
            <X className="size-4" />
          </button>
          <img
            src="/static/julienlucas.jpeg"
            alt="Julien Lucas"
            className="size-10 rounded-full border border-hairline object-cover"
          />
          <p className="display-sm mt-3">Call projet IA</p>
          <p className="mt-1 text-[0.8rem] leading-relaxed text-ink-muted">
            20 minutes pour valider votre projet d'automatisation ou d'application IA.
          </p>
          <Button variant="brand" size="sm" className="mt-3 w-full" onClick={goToContact}>
            <Calendar /> Prendre contact
          </Button>
        </div>
      ) : null}

      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="relative grid size-12 place-items-center rounded-full border border-white/15 bg-ink text-on-ink shadow-lift transition-colors hover:bg-[color-mix(in_srgb,var(--ink)_86%,var(--signal))]"
        aria-label="Contacter Julien"
        aria-expanded={open}
      >
        <span
          aria-hidden
          className="halo-spin pointer-events-none absolute -inset-0.5 rounded-full opacity-80"
          style={{
            filter: "blur(6px)",
            background:
              "conic-gradient(from 0deg, transparent 0deg, color-mix(in srgb, var(--signal) 25%, transparent) 120deg, var(--signal) 220deg, transparent 360deg)",
            WebkitMask: "radial-gradient(farthest-side, transparent calc(100% - 3px), #000 0)",
            mask: "radial-gradient(farthest-side, transparent calc(100% - 3px), #000 0)",
          }}
        />
        <Mail className="relative size-5" />
      </button>
    </div>
  );
}
