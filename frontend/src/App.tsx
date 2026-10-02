import { TooltipProvider } from "@/components/ui/tooltip";
import { Toaster } from "@/components/ui/sonner";
import { ContactWidget } from "@/components/site/ContactWidget";
import { Navbar } from "@/components/site/Navbar";
import { Footer } from "@/components/site/Footer";
import { Hero } from "@/components/sections/Hero";
import { Agents } from "@/components/sections/Agents";
import { Results } from "@/components/sections/Results";
import { Tools } from "@/components/sections/Tools";
import { AboutPage } from "@/components/pages/AboutPage";
import { useLenis } from "@/lib/use-lenis";

// Deux pages seulement : un routage par chemin suffit (Django sert index.html sur les deux).
const isAbout = window.location.pathname.replace(/\/+$/, "") === "/a-propos";

export default function App() {
  // Smooth scroll inertiel, comme sur Prospable.
  useLenis();

  return (
    <TooltipProvider delayDuration={150}>
      <Navbar />
      <main>
        {isAbout ? (
          <AboutPage />
        ) : (
          <>
            <Hero />
            <Agents />
            <Results />
            <Tools />
          </>
        )}
      </main>
      <Footer />
      <ContactWidget />
      <Toaster />
    </TooltipProvider>
  );
}
