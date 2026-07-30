export type Label = "CLEAN" | "OFFENSIVE" | "HATE";

export interface Membership {
  name: string;
  value: number;
  low: number;
  med: number;
  high: number;
}

export interface Rule {
  name: string;
  strength: number;
  conclusion: Label;
}

export interface Highlight {
  token: string;
  z: number;
}

export interface Prediction {
  label: Label;
  label_vn: string;
  color: string;
  probabilities: Record<Label, number>;
  p_mlp: Record<Label, number>;
  p_fuzzy: Record<Label, number>;
  lambda: number;
  memberships: Membership[];
  rules: Rule[];
  highlights: Highlight[];
  n_tokens: number;
  steps: Step[];
}

export interface FeatTerm { term: string; weight: number }
export interface TokZ { token: string; z: number }
export interface MuPair { label: string; value: number }
export interface Antecedent { label: string; value: number }
export interface RuleLive { name: string; antecedents: Antecedent[]; weight: number; strength: number; conclusion: Label }
export interface ClassScore { label: Label; score: number }
export interface LogitPair { label: Label; logit: number }
export interface NeuronAct { neuron: number; act: number }
export interface S1 { raw: string; normalized: string; n_tokens: number; tokens_sample: string[] }
export interface S2 { nnz: number; top_word: FeatTerm[]; top_char: FeatTerm[] }
export interface S3 { tokens_z: TokZ[]; crisp_raw: number[]; bounds_lo: number[]; bounds_hi: number[]; crisp_norm: number[] }
export interface S4 { mu: MuPair[] }
export interface S5 { rules: RuleLive[] }
export interface S6 { class_score: ClassScore[]; p_fuzzy: Record<Label, number> }
export interface S7 { h1_active: number; h1_top: NeuronAct[]; h2_active: number; logits: LogitPair[]; p_mlp: Record<Label, number> }
export interface S8 { lambda: number; final: Record<Label, number>; label: Label }
export type StepContent = S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8;
export interface Step { n: number; title: string; subtitle: string; content: StepContent }

export interface Sample {
  text: string;
  label: Label | "";
}

const API = "/api";

export async function predict(text: string): Promise<Prediction> {
  const r = await fetch(`${API}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (!r.ok) throw new Error(`API ${r.status}`);
  return r.json();
}

export async function sample(): Promise<Sample> {
  const r = await fetch(`${API}/sample`);
  return r.json();
}
