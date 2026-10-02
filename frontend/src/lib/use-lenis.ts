import { useEffect } from "react";
import Lenis from "lenis";
import "lenis/dist/lenis.css";

// Hauteur de la barre de navigation collante (h-14) + une marge : les ancres ne passent pas dessous.
const NAV_OFFSET = -72;

let instance: Lenis | null = null;

/** Défile vers une ancre (`#id`) ou en haut de page, avec Lenis s'il est monté. */
export function scrollToHash(hash: string) {
  if (instance) {
    instance.scrollTo(hash === "#top" ? 0 : hash, { offset: NAV_OFFSET });
    return;
  }
  document.querySelector(hash)?.scrollIntoView({ behavior: "smooth", block: "start" });
}

// Smooth scroll inertiel (Lenis), repris du projet Prospable, + interception des ancres
// internes avec un offset = hauteur de la nav. Les liens `/#id` ou `/a-propos#id` sont
// interceptés quand ils pointent vers la page courante ; sinon, navigation normale.
export function useLenis() {
  useEffect(() => {
    const lenis = new Lenis();
    instance = lenis;
    let rafId = 0;
    const raf = (time: number) => {
      lenis.raf(time);
      rafId = requestAnimationFrame(raf);
    };
    rafId = requestAnimationFrame(raf);

    // Arrivée depuis une autre page avec une ancre : Lenis a remplacé le scroll natif, donc le
    // saut d'ancre du navigateur ne se produit pas → on le refait une fois les dimensions
    // calculées (double rAF, sinon `limit` vaut encore 0 et le scroll serait bloqué en haut).
    const initialHash = window.location.hash;
    if (initialHash.length > 1) {
      requestAnimationFrame(() =>
        requestAnimationFrame(() =>
          lenis.scrollTo(initialHash, { offset: NAV_OFFSET, immediate: true, force: true }),
        ),
      );
    }

    const onClick = (e: MouseEvent) => {
      if (e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
      const anchor = (e.target as HTMLElement).closest<HTMLAnchorElement>("a[href*='#']");
      if (!anchor || anchor.target === "_blank") return;
      const url = new URL(anchor.href, window.location.href);
      const samePage =
        url.origin === window.location.origin &&
        url.pathname.replace(/\/+$/, "") === window.location.pathname.replace(/\/+$/, "");
      if (!samePage || url.hash.length < 2) return;
      e.preventDefault();
      scrollToHash(url.hash);
    };
    document.addEventListener("click", onClick);

    return () => {
      document.removeEventListener("click", onClick);
      cancelAnimationFrame(rafId);
      lenis.destroy();
      instance = null;
    };
  }, []);
}
