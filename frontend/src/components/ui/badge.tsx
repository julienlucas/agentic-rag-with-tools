import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";

import { cn } from "@/lib/utils";

/*
 * Badges au même angle que les boutons (2px). La variante `mono` est l'étiquette
 * technique du système : Geist Mono en capitales, filet, pas d'aplat — c'est elle
 * qui porte les noms de modèles et d'outils.
 */
const badgeVariants = cva(
  "inline-flex items-center gap-1.5 rounded-sm border px-2 py-0.5 text-xs font-medium leading-5 transition-colors focus:outline-none",
  {
    variants: {
      variant: {
        default: "border-transparent bg-ink text-on-ink",
        brand: "border-transparent bg-brand-surface text-brand-deep",
        sand: "border-transparent bg-paper-2 text-ink-muted",
        secondary: "border-transparent bg-paper-2 text-ink-soft",
        success: "border-transparent bg-success/10 text-success",
        warning: "border-transparent bg-warning/12 text-warning",
        destructive: "border-transparent bg-destructive/10 text-destructive",
        outline: "border-hairline-strong text-ink",
        mono: "mono-xs border-hairline bg-transparent px-2 py-1 uppercase tracking-[0.06em] text-ink-muted",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  },
);

export interface BadgeProps
  extends React.HTMLAttributes<HTMLDivElement>, VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, ...props }: BadgeProps) {
  return <div className={cn(badgeVariants({ variant }), className)} {...props} />;
}

export { Badge, badgeVariants };
