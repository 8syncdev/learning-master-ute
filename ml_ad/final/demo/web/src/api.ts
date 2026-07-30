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
}

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
