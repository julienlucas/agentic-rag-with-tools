import { useCallback, useEffect, useLayoutEffect, useRef, useState, type ReactNode } from "react";
import { ArrowUp, ChevronDown, Clock, RotateCcw, ShieldAlert, Sparkles } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";
import { api, type Citation } from "@/api";
import { renderMarkdown } from "@/lib/markdown";
import { parseVerificationReport } from "@/lib/verification";
import { useElapsed } from "@/hooks/useTimer";
import { CorpusPanel } from "./CorpusPanel";
import { exampleDocs, exampleOutput, type ExampleDoc } from "./examples";
import { pipelineSteps } from "./pipeline";
import type { DocState, Turn } from "./types";

const DEFAULT_EXAMPLE = exampleDocs[0];
const EXAMPLE_SIGNALS = parseVerificationReport(exampleOutput.report);
const MAX_ANSWER_H = 130;

/** Réponse repliée à 130 px, dépliable par « Voir la suite ». */
function CollapsibleAnswer({ children }: { children: ReactNode }) {
  const [open, setOpen] = useState(false);
  const [overflows, setOverflows] = useState(false);
  const [fullHeight, setFullHeight] = useState(0);
  const inner = useRef<HTMLDivElement>(null);

  /* mesure avant peinture : sinon le bloc s'affiche déplié puis se referme */
  useLayoutEffect(() => {
    const el = inner.current;
    if (!el) return;
    const check = () => {
      setFullHeight(el.scrollHeight);
      setOverflows(el.scrollHeight > MAX_ANSWER_H + 8);
    };
    check();
    const ro = new ResizeObserver(check);
    ro.observe(el);
    return () => ro.disconnect();
  }, [children]);

  const collapsed = !fullHeight || (overflows && !open);

  return (
    <div>
      <div
        className="relative overflow-hidden transition-[max-height] duration-300"
        style={{ maxHeight: collapsed ? MAX_ANSWER_H : fullHeight }}
      >
        <div ref={inner} className="!text-xs !leading-relaxed !text-ink-muted">
          {children}
        </div>
        {overflows && !open ? (
          <span
            aria-hidden
            className="pointer-events-none absolute inset-x-0 bottom-0 h-10 bg-gradient-to-t from-paper-2 to-transparent"
          />
        ) : null}
      </div>
      {overflows ? (
        <button
          type="button"
          onClick={() => setOpen((o) => !o)}
          className="mono-xs mt-2 inline-flex items-center gap-1 text-brand-deep transition-colors hover:text-brand-strong cursor-pointer"
        >
          {open ? "Réduire" : "Voir la suite"}
          <ChevronDown
            className={cn("size-3 transition-transform", open && "rotate-180")}
          />
        </button>
      ) : null}
    </div>
  );
}

/** Marqueur de citation [n] : renvoie au passage transmis au modèle. */
function CiteMark({ n, citation }: { n: number; citation?: Citation }) {
  const mark = (
    <sup className="mono-xs ml-0.5 cursor-help rounded bg-brand-surface-strong px-1 py-px text-[0.6rem] text-brand-deep">
      {n}
    </sup>
  );
  if (!citation) return mark;
  return (
    <Tooltip>
      <TooltipTrigger asChild>{mark}</TooltipTrigger>
      <TooltipContent className="max-w-sm">
        <span className="mono-xs block text-brand-light">{citation.locator}</span>
        <span className="mt-1 block text-xs leading-relaxed">{citation.excerpt}</span>
      </TooltipContent>
    </Tooltip>
  );
}

/** Liste des passages réellement cités dans la réponse. */
function CitedSources({ answer, citations }: { answer: string; citations: Citation[] }) {
  const used = Array.from(new Set([...answer.matchAll(/\[(\d+)\]/g)].map((m) => Number(m[1]))))
    .sort((a, b) => a - b)
    .map((n) => citations.find((c) => c.n === n))
    .filter((c): c is Citation => Boolean(c));
  if (!used.length) return null;
  return (
    <div className="mt-3 border-t border-hairline pt-3">
      <span className="eyebrow">Passages cités</span>
      <ul className="mt-2 space-y-1.5">
        {used.map((c) => (
          <li key={c.n} className="flex gap-2 text-[0.7rem] leading-relaxed text-muted-foreground">
            <span className="mono-xs shrink-0 text-brand-deep">[{c.n}]</span>
            <span className="min-w-0 flex-1">
              <span className="font-medium text-ink-muted">{c.locator}</span> — {c.excerpt}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}

/** Marqueur de tour : Q pour la question, R pour la réponse. */
function Marker({ kind }: { kind: "q" | "r" }) {
  return (
    <span
      aria-hidden
      className={cn(
        "mt-[0.09rem] grid size-5 shrink-0 place-items-center rounded font-mono text-[0.7rem] font-semibold",
        kind === "q"
          ? "text-brand-deep"
          : "text-success",
      )}
    >
      {kind === "q" ? "Q" : "R"}
    </span>
  );
}

function docFromExample(ex: ExampleDoc): DocState {
  return {
    id: ex.id,
    title: ex.title,
    fileName: ex.fileName,
    kind: ex.kind,
    pages: ex.pages,
    type: ex.type,
    description: ex.description,
    source: "example",
    status: "loading",
  };
}

function docFromFile(file: File): DocState {
  return {
    id: `upload-${Date.now()}`,
    title: file.name,
    fileName: file.name,
    kind: (file.name.split(".").pop() ?? "DOC").toUpperCase(),
    type: "Document utilisateur",
    source: "upload",
    file,
    status: "loading",
  };
}


export function DemoChat({ children }: { children?: ReactNode }) {
  const [sessionId] = useState(() => `session_${Date.now().toString(36)}`);
  const [doc, setDoc] = useState<DocState | null>(null);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [draft, setDraft] = useState("");
  const [pending, setPending] = useState(false);
  const [stage, setStage] = useState(0);
  const elapsed = useElapsed(pending);
  const timers = useRef<ReturnType<typeof setTimeout>[]>([]);
  const viewport = useRef<HTMLDivElement>(null);
  const loadToken = useRef(0);

  const clearTimers = () => {
    timers.current.forEach(clearTimeout);
    timers.current = [];
  };
  useEffect(() => clearTimers, []);

  useEffect(() => {
    if (turns.length === 0 && !pending) return;
    const el = viewport.current;
    if (el) el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
  }, [turns, pending, stage]);

  /* ---------- chargement / indexation d'un document ---------- */

  const indexDoc = useCallback(
    async (next: DocState) => {
      const token = ++loadToken.current;
      setDoc({ ...next, status: "loading", chunks: undefined, error: undefined });
      setTurns([]);
      const started = Date.now();
      try {
        const res =
          next.source === "upload" && next.file
            ? await api.uploadFile(next.file, sessionId)
            : await api.loadFile(next.fileName, sessionId);
        if (token !== loadToken.current) return;
        setDoc((d) =>
          d && d.id === next.id
            ? {
                ...d,
                status: "ready",
                chunks: res.chunks_count,
                loadSeconds: Math.max(1, Math.round((Date.now() - started) / 1000)),
              }
            : d,
        );
      } catch (err) {
        if (token !== loadToken.current) return;
        const message = err instanceof Error ? err.message : "Erreur de connexion au backend";
        setDoc((d) => (d && d.id === next.id ? { ...d, status: "error", error: message } : d));
        toast.error("Indexation impossible", { description: message });
      }
    },
    [sessionId],
  );

  useEffect(() => {
    indexDoc(docFromExample(DEFAULT_EXAMPLE));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const selectExample = (id: string) => {
    const ex = exampleDocs.find((e) => e.id === id);
    if (ex && ex.id !== doc?.id) indexDoc(docFromExample(ex));
  };

  const upload = (file: File) => indexDoc(docFromFile(file));

  const removeUpload = () => indexDoc(docFromExample(DEFAULT_EXAMPLE));

  const retry = () => {
    if (doc) indexDoc(doc);
  };

  /* ---------- question ---------- */

  const ready = doc?.status === "ready";
  const busy = pending || doc?.status === "loading";

  async function ask(question: string) {
    const q = question.trim();
    if (!q || !ready || pending) return;

    clearTimers();
    setDraft("");
    setPending(true);
    setStage(0);
    setTurns((t) => [...t, { id: `u-${Date.now()}`, role: "user", text: q }]);

    // Animation estimée de la trace (le backend ne streame pas).
    let acc = 0;
    pipelineSteps.forEach((s, i) => {
      if (s.ms > 0 && i + 1 < pipelineSteps.length) {
        acc += s.ms;
        const nextIndex = pipelineSteps[i + 1].conditional ? i + 2 : i + 1;
        timers.current.push(setTimeout(() => setStage(nextIndex), acc));
      }
    });

    const started = Date.now();
    try {
      const res = await api.processQuestion(q, sessionId);
      const signals = parseVerificationReport(res.verification_report);
      setTurns((t) => [
        ...t,
        {
          id: `a-${Date.now()}`,
          role: "assistant",
          answer: res.draft_answer,
          signals,
          elapsed: Math.round((Date.now() - started) / 1000),
          citations: res.citations ?? [],
        },
      ]);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Erreur de connexion au backend";
      const signals = parseVerificationReport(`Erreur: ${message}`);
      setTurns((t) => [
        ...t,
        {
          id: `a-${Date.now()}`,
          role: "assistant",
          answer: message,
          signals,
          elapsed: Math.round((Date.now() - started) / 1000),
          citations: [],
          failed: true,
        },
      ]);
      toast.error("Le pipeline a échoué", { description: message });
    } finally {
      clearTimers();
      setPending(false);
      setStage(pipelineSteps.length);
    }
  }

  const reset = () => {
    clearTimers();
    setTurns([]);
    setPending(false);
    setDraft("");
  };

  const activeExample = doc?.source === "example" ? exampleDocs.find((e) => e.id === doc.id) : undefined;
  const suggestions = activeExample?.questions ?? [];
  const showExample = turns.length === 0 && !pending && doc?.id === exampleOutput.docId;

  return (
    <div>
      <CorpusPanel
        active={doc}
        onSelectExample={selectExample}
        onUpload={upload}
        onRemoveUpload={removeUpload}
        onRetry={retry}
        busy={busy}
      />

      {children}

      <h3 className="-mt-10 display-sm py-6 pb-3">
        La réponse
      </h3>

      <div className="card-paper bg-white overflow-hidden">
        <div>
          {/* conversation */}
          <div className="flex min-h-[28rem] flex-col">
            <ScrollArea className="h-fit flex-1" viewportRef={viewport}>
              <div className="space-y-5 p-5">
                {showExample ? (
                  <div className="rise-in space-y-5">
                    {doc?.status === "loading" ? (
                      <span className="mono-xs text-muted-foreground">
                        indexation en cours…
                      </span>
                    ) : null}
                    <div className="flex justify-end">
                      <p className="flex max-w-[85%] gap-2.5 rounded-2xl bg-ink px-4 py-2.5 text-sm leading-relaxed text-on-ink">
                        <Marker kind="q" />
                        <span className="min-w-0 flex-1">
                          {exampleOutput.question}
                        </span>
                      </p>
                    </div>
                    <div className="max-w-[95%]">
                      <div className="min-w-0 flex-1">
                        <div className="mb-2 flex flex-wrap items-center gap-2">
                          <span className="mono-xs text-muted-foreground">
                            confiance reranker{" "}
                            {EXAMPLE_SIGNALS.rerankScore?.toFixed(2)}
                          </span>
                          <span className="mono-xs inline-flex items-center gap-1 text-muted-foreground">
                            <Clock className="size-3" /> {exampleOutput.elapsed}
                            s
                          </span>
                        </div>
                        <div className="flex gap-2.5 rounded-2xl border border-hairline bg-paper-2 px-4 py-3.5">
                          <Marker kind="r" />
                          <div className="min-w-0 flex-1">
                            <CollapsibleAnswer>
                              <div className="prose-rag">
                                {renderMarkdown(exampleOutput.answer)}
                              </div>
                            </CollapsibleAnswer>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                ) : null}

                {turns.length === 0 && !pending && !showExample ? (
                  <div className="rise-in">
                    <h3 className="display-sm">
                      {doc
                        ? `Interrogez « ${doc.title} ».`
                        : "Chargez un document pour commencer."}
                    </h3>
                    <p className="copy mt-2 text-sm">
                      Le pipeline cible le document, cherche en lexical et en
                      vectoriel, fusionne, reranke, vérifie la pertinence des
                      passages, corrige la recherche si besoin — puis répond
                      uniquement à partir des preuves retenues.
                    </p>
                    {doc?.status === "loading" ? (
                      <p className="mono-xs mt-4 text-muted-foreground">
                        Mistral OCR et indexation en cours… la première question
                        sera possible dans quelques secondes.
                      </p>
                    ) : null}
                    {suggestions.length ? (
                      <div className="mt-5 grid gap-2">
                        {suggestions.map((s) => (
                          <button
                            key={s.text}
                            type="button"
                            disabled={!ready}
                            onClick={() => ask(s.text)}
                            className="group flex items-start gap-3 rounded-md border border-hairline bg-paper px-3.5 py-3 text-left text-sm transition-colors hover:border-brand hover:bg-brand-surface disabled:cursor-not-allowed disabled:opacity-50"
                          >
                            <Sparkles className="mt-0.5 size-3.5 shrink-0 text-brand-deep" />
                            <span className="min-w-0 flex-1">
                              <span className="block leading-snug">
                                {s.text}
                              </span>
                              <span className="mono-xs mt-1 block text-muted-foreground">
                                {s.hint}
                              </span>
                            </span>
                            <ArrowUp className="mt-0.5 size-3.5 shrink-0 rotate-45 text-muted-foreground transition-transform group-hover:translate-x-0.5" />
                          </button>
                        ))}
                      </div>
                    ) : null}
                  </div>
                ) : null}

                {turns.map((turn) =>
                  turn.role === "user" ? (
                    <div key={turn.id} className="flex justify-end">
                      <p className="rise-in flex max-w-[85%] gap-2.5 rounded-2xl rounded-br-sm bg-ink px-4 py-2.5 text-sm leading-relaxed text-on-ink">
                        <Marker kind="q" />
                        <span className="min-w-0 flex-1">{turn.text}</span>
                      </p>
                    </div>
                  ) : (
                    <div key={turn.id} className="rise-in max-w-[95%]">
                      <div className="min-w-0 flex-1">
                        <div className="mb-2 flex flex-wrap items-center gap-2">
                          {turn.signals.rerankScore != null ? (
                            <span className="mono-xs text-muted-foreground">
                              confiance reranker{" "}
                              {turn.signals.rerankScore.toFixed(2)}
                            </span>
                          ) : null}
                          <span className="mono-xs inline-flex items-center gap-1 text-muted-foreground">
                            <Clock className="size-3" /> {turn.elapsed}s
                          </span>
                        </div>
                        <div
                          className={cn(
                            "flex gap-2.5 rounded-2xl border px-4 py-3.5",
                            turn.failed
                              ? "border-destructive/40 bg-destructive/5"
                              : "border-hairline bg-paper-2",
                          )}
                        >
                          {turn.failed ? null : <Marker kind="r" />}
                          <div className="min-w-0 flex-1">
                            {turn.failed ? (
                              <p className="flex gap-2 text-sm text-destructive">
                                <ShieldAlert className="mt-0.5 size-4 shrink-0" />
                                {turn.answer}
                              </p>
                            ) : (
                              <CollapsibleAnswer>
                                <div className="prose-rag">
                                  {renderMarkdown(turn.answer, (n) => (
                                    <CiteMark n={n} citation={turn.citations.find((c) => c.n === n)} />
                                  ))}
                                </div>
                              </CollapsibleAnswer>
                            )}
                            {!turn.failed ? (
                              <CitedSources answer={turn.answer} citations={turn.citations} />
                            ) : null}
                            {!turn.failed &&
                            turn.signals.correctiveRounds > 0 ? (
                              <p className="mt-3 flex gap-2 border-t border-hairline pt-3 text-xs text-muted-foreground">
                                <Sparkles className="mt-0.5 size-3.5 shrink-0 text-brand-deep" />
                                Le modèle a jugé le contexte insuffisant et a
                                cherché lui-même (search, grep, read_page) avant
                                de répondre — le détail est dans le rapport.
                              </p>
                            ) : null}
                          </div>
                        </div>
                      </div>
                    </div>
                  ),
                )}

                {pending ? (
                  <div className="rise-in space-y-2 rounded-2xl border border-dashed border-brand/50 bg-brand-surface/40 px-4 py-3.5">
                    {pipelineSteps
                      .filter((s) => !s.conditional)
                      .map((s) => {
                        const i = pipelineSteps.indexOf(s);
                        const done = stage > i;
                        const running = stage === i;
                        return (
                          <div
                            key={s.key}
                            className={cn(
                              "flex items-center gap-2.5 text-xs transition-opacity",
                              !done && !running && "opacity-40",
                            )}
                          >
                            {done ? (
                              <span className="size-3.5 shrink-0 rounded-full bg-success/80" />
                            ) : (
                              <span
                                className={cn(
                                  "size-3.5 shrink-0 rounded-full border border-brand",
                                  running && "signal-dot bg-brand",
                                )}
                              />
                            )}
                            <span
                              className={cn(
                                "font-medium",
                                done && "text-ink-muted",
                              )}
                            >
                              {s.label}
                            </span>
                            {running ? (
                              <span className="mono-xs text-muted-foreground">
                                en cours…
                              </span>
                            ) : null}
                          </div>
                        );
                      })}
                    <p className="mono-xs pt-1 text-muted-foreground">
                      {elapsed}s · 3 à 5 appels LLM en série, comptez 10 à 40 s
                    </p>
                  </div>
                ) : null}
              </div>
            </ScrollArea>

            {/* saisie */}
            <div className="border-t border-hairline p-3 px-5">
              <div className="flex items-end gap-2 rounded-sm border border-hairline-strong bg-paper p-2 transition-colors focus-within:border-brand">
                <textarea
                    rows={1}
                  value={draft}
                  onChange={(e) => setDraft(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      ask(draft);
                    }
                  }}
                  disabled={!ready || pending}
                  placeholder={
                    ready
                      ? "Posez une question sur le document…"
                      : doc?.status === "loading"
                        ? "Indexation en cours…"
                        : "Chargez un document pour poser une question"
                  }
                  className="max-h-32 min-h-9 flex-1 resize-none bg-transparent px-2 py-1.5 text-sm outline-none placeholder:text-muted-foreground disabled:cursor-not-allowed"
                />
                <Button
                  variant="brand"
                  onClick={() => ask(draft)}
                  disabled={!draft.trim() || !ready || pending}
                  className="h-9 shrink-0 px-3.5"
                >
                  Envoyer
                  <ArrowUp />
                </Button>
              </div>
              {suggestions.length || turns.length ? (
                <div className="mt-2 flex flex-wrap items-center gap-1.5">
                  {suggestions.length ? (
                    <span className="eyebrow mr-1">Questions</span>
                  ) : null}
                  {turns.length ? (
                    <button
                      type="button"
                      onClick={reset}
                      disabled={pending}
                      className="mono-xs inline-flex items-center gap-1 rounded-full border border-hairline px-2.5 py-1 text-muted-foreground transition-colors hover:border-brand hover:text-brand-deep disabled:opacity-40"
                    >
                      <RotateCcw className="size-3" /> Réinitialiser
                    </button>
                  ) : null}
                  {suggestions.map((s) => (
                    <button
                      key={s.text}
                      type="button"
                      disabled={!ready || pending}
                      onClick={() => ask(s.text)}
                      className="mono-xs cursor-pointer max-w-full truncate rounded-full border border-hairline px-2.5 py-1 text-muted-foreground transition-colors hover:border-brand hover:text-brand-deep disabled:opacity-40"
                      title={s.text}
                    >
                      {s.text.slice(0, 48)}…
                    </button>
                  ))}
                </div>
              ) : null}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
