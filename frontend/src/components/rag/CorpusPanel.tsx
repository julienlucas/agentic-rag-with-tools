import { useCallback, useRef, useState, type DragEvent } from "react";
import { Check, FileText, Loader2, RefreshCw, TriangleAlert, Upload, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { useElapsed } from "@/hooks/useTimer";
import { exampleDocs, MAX_FILE_MB, SUPPORTED_EXTENSIONS } from "./examples";
import type { DocState } from "./types";

/** Une puce de document : exemple sélectionnable ou fichier uploadé. */
function DocChip({
  doc,
  active,
  onSelect,
  onRemove,
  onRetry,
}: {
  doc: DocState;
  active: boolean;
  onSelect?: () => void;
  onRemove?: () => void;
  onRetry?: () => void;
}) {
  const elapsed = useElapsed(doc.status === "loading");
  const clickable = !!onSelect && !active;

  return (
    <span
      role={clickable ? "button" : undefined}
      tabIndex={clickable ? 0 : undefined}
      onClick={clickable ? onSelect : undefined}
      onKeyDown={
        clickable
          ? (e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                onSelect?.();
              }
            }
          : undefined
      }
      className={cn(
        "inline-flex max-w-full items-center gap-2 rounded-full border px-3 py-1.5 text-xs transition-colors",
        active ? "border-brand bg-brand-surface text-ink" : "border-hairline bg-paper text-ink-muted",
        clickable && "cursor-pointer hover:border-brand hover:text-ink",
      )}
    >
      {doc.status === "loading" ? (
        <Loader2 className="size-3.5 shrink-0 animate-spin text-brand-deep" />
      ) : doc.status === "error" ? (
        <TriangleAlert className="size-3.5 shrink-0 text-destructive" />
      ) : active ? (
        <Check className="size-3.5 shrink-0 text-brand-deep" />
      ) : (
        <FileText className="size-3.5 shrink-0 text-muted-foreground" />
      )}

      <span className="min-w-0 truncate font-medium">{doc.title}</span>

      {doc.status === "loading" ? (
        <span className="meta">{elapsed}s</span>
      ) : doc.status === "error" && onRetry ? (
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            onRetry();
          }}
          className="mono-xs inline-flex items-center gap-1 text-destructive hover:underline"
        >
          <RefreshCw className="size-3" /> réessayer
        </button>
      ) : (
        <span className="meta shrink-0 whitespace-nowrap">
          {doc.pages ? `${doc.pages} p.` : doc.kind}
        </span>
      )}

      {onRemove ? (
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            onRemove();
          }}
          aria-label={`Retirer ${doc.title}`}
          className="text-muted-foreground transition-colors hover:text-destructive"
        >
          <X className="size-3.5" />
        </button>
      ) : null}
    </span>
  );
}

export function CorpusPanel({
  active,
  onSelectExample,
  onUpload,
  onRemoveUpload,
  onRetry,
  busy,
}: {
  active: DocState | null;
  onSelectExample: (id: string) => void;
  onUpload: (file: File) => void;
  onRemoveUpload: () => void;
  onRetry: () => void;
  busy: boolean;
}) {
  const fileInput = useRef<HTMLInputElement>(null);
  const [fileError, setFileError] = useState<string | null>(null);

  const accept = useCallback(
    (files: FileList | null) => {
      setFileError(null);
      const file = files?.[0];
      if (!file) return;
      if (!SUPPORTED_EXTENSIONS.some((ext) => file.name.toLowerCase().endsWith(ext))) {
        setFileError("Format non supporté : PDF, DOCX, TXT ou MD.");
        return;
      }
      if (file.size > MAX_FILE_MB * 1024 * 1024) {
        setFileError(`Fichier trop lourd (max ${MAX_FILE_MB} MB).`);
        return;
      }
      onUpload(file);
    },
    [onUpload],
  );

  const examples: DocState[] = exampleDocs.map((ex) =>
    active?.id === ex.id
      ? active
      : {
          id: ex.id,
          title: ex.title,
          fileName: ex.fileName,
          kind: ex.kind,
          pages: ex.pages,
          type: ex.type,
          description: ex.description,
          source: "example",
          status: "idle",
        },
  );

  // Compteur d'entrées : dragenter/dragleave se déclenchent aussi sur les enfants de la zone.
  const dragDepth = useRef(0);
  const [dragging, setDragging] = useState(false);

  const onDragEnter = (e: DragEvent) => {
    if (!e.dataTransfer.types.includes("Files")) return;
    e.preventDefault();
    dragDepth.current += 1;
    setDragging(true);
  };
  const onDragLeave = (e: DragEvent) => {
    e.preventDefault();
    dragDepth.current = Math.max(0, dragDepth.current - 1);
    if (dragDepth.current === 0) setDragging(false);
  };
  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    dragDepth.current = 0;
    setDragging(false);
    if (!busy) accept(e.dataTransfer.files);
  };

  return (
    // Pleine hauteur de la conversation : la zone de dépôt s'étire, les exemples restent en bas.
    <div className="flex flex-col gap-6 lg:flex-1">
      <input
        ref={fileInput}
        type="file"
        accept={SUPPORTED_EXTENSIONS.join(",")}
        className="hidden"
        onChange={(e) => {
          accept(e.target.files);
          e.target.value = "";
        }}
      />

      {/* Zone de dépôt : glisser un fichier, ou cliquer le bouton au centre. */}
      <div
        onDragEnter={onDragEnter}
        onDragOver={(e) => e.preventDefault()}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
        className={cn(
          "flex flex-col items-center justify-center gap-3 rounded-[var(--radius-media)] border border-dashed px-4 py-8 text-center transition-colors lg:flex-1",
          dragging ? "border-brand bg-brand-surface" : "border-brand bg-paper",
          busy && "opacity-60",
        )}
      >
        <span className="grid size-10 place-items-center rounded-full bg-brand-surface text-brand [&_svg]:size-4.5">
          <Upload />
        </span>
        <div>
          <p className="text-sm font-medium text-ink">
            {dragging ? "Déposez le document" : "Glissez un document ici"}
          </p>
          <p className="meta mt-1">
            PDF, DOCX, TXT ou MD · {MAX_FILE_MB} MB max
          </p>
        </div>
        <Button
          variant="outline"
          size="sm"
          disabled={busy}
          onClick={() => fileInput.current?.click()}
        >
          Charger un document
        </Button>
      </div>

      {fileError ? <p className="text-xs text-destructive">{fileError}</p> : null}

      {active?.source === "upload" ? (
        <DocChip
          doc={active}
          active
          onRemove={busy ? undefined : onRemoveUpload}
          onRetry={onRetry}
        />
      ) : null}

      <div>
        <span className="meta">ou parmi nos exemples</span>
        <div className="mt-3 flex flex-col items-start gap-2">
          {examples.map((d) => (
            <DocChip
              key={d.id}
              doc={d}
              active={active?.id === d.id}
              onSelect={busy ? undefined : () => onSelectExample(d.id)}
              onRetry={onRetry}
            />
          ))}
        </div>
      </div>
    </div>
  );
}
