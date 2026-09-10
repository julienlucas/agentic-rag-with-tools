import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

export function Container({
  children,
  className,
  width = "default",
}: {
  children: ReactNode;
  className?: string;
  width?: "default" | "narrow" | "wide";
}) {
  return (
    <div
      className={cn(
        "mx-auto w-full px-[var(--gutter)]",
        width === "narrow" && "max-w-3xl",
        width === "default" && "max-w-6xl",
        width === "wide" && "max-w-7xl",
        className,
      )}
    >
      {children}
    </div>
  );
}

/**
 * Surtitre : Geist Mono en capitales espacées, sans pastille ni aplat — le contraste
 * de casse et de chasse suffit à le détacher du titre qui suit.
 */
export function Eyebrow({ children, className }: { children: ReactNode; className?: string }) {
  return <span className={cn("eyebrow", className)}>{children}</span>;
}

/** Section numérotée, mise en page éditoriale : index en rail, surtitre mono, titre display. */
export function Section({
  index,
  eyebrow,
  title,
  intro,
  children,
  className,
  tone = "paper",
  id,
}: {
  index?: string;
  eyebrow?: string;
  title?: ReactNode;
  intro?: ReactNode;
  children?: ReactNode;
  className?: string;
  tone?: "paper" | "muted" | "ink";
  id?: string;
}) {
  const onInk = tone === "ink";
  return (
    <section
      id={id}
      className={cn(
        "scroll-mt-16 border-t border-hairline py-[var(--section-y)]",
        tone === "muted" && "bg-paper-2",
        onInk && "surface-ink grain bg-ink",
        className,
      )}
    >
      <Container>
        <div className="grid gap-x-10 gap-y-8 lg:grid-cols-[6rem_1fr]">
          {/* Rail de gauche : le numéro de section, aligné sur la première ligne du titre. */}
          <div className="hidden lg:block">
            {index ? (
              <div className="sticky top-8">
                <span className="mono-xs block tracking-[0.08em] text-ink-faint">
                  ({index})
                </span>
                <span
                  aria-hidden
                  className={cn("mt-3 block h-px w-8", onInk ? "bg-gold-300" : "bg-brand")}
                />
              </div>
            ) : null}
          </div>

          <div className="min-w-0">
            {eyebrow ? (
              <Eyebrow>{eyebrow}</Eyebrow>
            ) : null}
            {title ? (
              <h2 className="display-lg mt-4 max-w-3xl">{title}</h2>
            ) : null}
            {intro ? (
              <div className="copy mt-5">{intro}</div>
            ) : null}
            {children ? <div className="mt-12">{children}</div> : null}
          </div>
        </div>
      </Container>
    </section>
  );
}
