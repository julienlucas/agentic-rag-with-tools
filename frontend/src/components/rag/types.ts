import type { VerificationSignals } from "@/lib/verification";
import type { Citation } from "@/api";
import type { Evidence } from "./examples";

export type DocState = {
  id: string;
  title: string;
  fileName: string;
  kind: string;
  pages?: number;
  type: string;
  description?: string;
  source: "example" | "upload";
  file?: File;
  status: "idle" | "loading" | "ready" | "error";
  chunks?: number;
  loadSeconds?: number;
  error?: string;
};

export type Turn =
  | { id: string; role: "user"; text: string }
  | {
      id: string;
      role: "assistant";
      answer: string;
      signals: VerificationSignals;
      elapsed: number;
      citations: Citation[];
      evidence?: Evidence[];
      failed?: boolean;
      feedbackToken?: string | null;
    };
