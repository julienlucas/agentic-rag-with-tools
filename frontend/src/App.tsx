import { TooltipProvider } from "@/components/ui/tooltip";
import { Toaster } from "@/components/ui/sonner";
import { ContactWidget } from "@/components/site/ContactWidget";
import { Hero } from "@/components/sections/Hero";
import { Agents } from "@/components/sections/Agents";
import { Results } from "@/components/sections/Results";
import { Tools } from "@/components/sections/Tools";
import { Contact } from "@/components/sections/Contact";

export default function App() {
  return (
    <TooltipProvider delayDuration={150}>
      <main>
        <Hero />
        <Agents />
        <Results />
        <Tools />
        <Contact />
      </main>
      <ContactWidget />
      <Toaster />
    </TooltipProvider>
  );
}
