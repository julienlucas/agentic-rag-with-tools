import * as React from "react";
import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";

import { cn } from "@/lib/utils";

/*
 * Boutons « Papier & Signal ».
 *
 * Angle quasi vif (rounded-sm = 2px), filet de 1px sur toutes les variantes — y compris
 * les aplats — pour que pleins et contours s'alignent au pixel près quand ils sont côte
 * à côte. Pas d'ombre : la hiérarchie passe par la valeur, pas par l'élévation.
 * Sur le primaire, le hover ne fonce pas l'encre, il y infuse 14 % d'or.
 * Toute icône placée en fin de bouton glisse de 3px au survol (`[&_svg:last-child]`).
 */
const buttonVariants = cva(
  [
    "group/btn inline-flex cursor-pointer items-center justify-center gap-2 whitespace-nowrap",
    "rounded-sm border border-transparent font-medium tracking-[-0.005em] leading-none",
    "transition-[background-color,border-color,color] duration-[var(--dur-fast)] ease-[var(--ease-out)]",
    "focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring",
    "disabled:pointer-events-none disabled:opacity-45 disabled:cursor-not-allowed",
    "[&_svg]:pointer-events-none [&_svg]:size-4 [&_svg]:shrink-0",
    "[&_svg:last-child]:transition-transform [&_svg:last-child]:duration-[var(--dur-base)] [&_svg:last-child]:ease-[var(--ease-out)]",
    "hover:[&_svg:last-child]:translate-x-[3px] focus-visible:[&_svg:last-child]:translate-x-[3px]",
  ],
  {
    variants: {
      variant: {
        default:
          "bg-ink text-on-ink border-ink hover:bg-[color-mix(in_srgb,var(--ink)_86%,var(--signal))] hover:border-[color-mix(in_srgb,var(--ink)_86%,var(--signal))]",
        brand:
          "bg-brand text-on-ink border-brand hover:bg-brand-strong hover:border-brand-strong",
        ink: "bg-ink text-on-ink border-ink hover:bg-ink-soft hover:border-ink-soft",
        destructive:
          "bg-destructive text-destructive-foreground border-destructive hover:bg-[color-mix(in_srgb,var(--destructive)_88%,black)]",
        outline:
          "border-hairline-strong bg-transparent text-ink hover:border-ink hover:bg-paper-2/60",
        onInk:
          "border-white/25 bg-transparent text-on-ink hover:border-white/70 hover:bg-white/8",
        secondary:
          "bg-paper-2 text-ink border-transparent hover:bg-paper-3",
        ghost:
          "text-ink hover:underline hover:decoration-1 hover:underline-offset-4",
        link: "text-brand-deep underline-offset-4 hover:underline",
      },
      size: {
        default: "min-h-11 px-5 py-3 text-[0.9375rem]",
        sm: "min-h-9 px-3.5 py-2 text-[0.8125rem]",
        lg: "min-h-12 px-6 py-3.5 text-[0.9375rem]",
        icon: "size-9 p-0",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  },
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>, VariantProps<typeof buttonVariants> {
  asChild?: boolean;
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : "button";
    return (
      <Comp className={cn(buttonVariants({ variant, size, className }))} ref={ref} {...props} />
    );
  },
);
Button.displayName = "Button";

export { Button, buttonVariants };
