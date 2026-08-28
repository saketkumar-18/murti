// Murti browser tokenizer — mirrors murti/text.py exactly.
// Vocabulary order must match Python: SPECIAL + STRUCT + SHAPES + ADJECTIVES.
export const WORDS = [
  "<pad>", "<unk>",
  "a", "on", "top", "of", "next", "to", "and", "two", "three", "with",
  "sphere", "cube", "cylinder", "cone", "torus", "capsule", "pyramid",
  "prism", "star", "arch", "cross", "table", "chair", "snowman", "rocket",
  "tree", "dumbbell", "block",
  "tall", "wide", "flat", "small", "large", "thin", "chunky", "slender",
  "squat", "stretched",
];
export const VOCAB = Object.fromEntries(WORDS.map((w, i) => [w, i]));
export const VOCAB_SIZE = WORDS.length; // 40
export const PAD_ID = 0;
export const UNK_ID = 1;
export const MAX_LEN = 12;

export function tokenize(caption) {
  const ids = [];
  for (const raw of caption.toLowerCase().replace(/,/g, " ").split(/\s+/)) {
    if (!raw) continue;
    let wid = VOCAB[raw];
    if (wid === undefined && raw.endsWith("s")) wid = VOCAB[raw.slice(0, -1)];
    ids.push(wid === undefined ? UNK_ID : wid);
  }
  const out = ids.slice(0, MAX_LEN);
  while (out.length < MAX_LEN) out.push(PAD_ID);
  return out;
}

export function decode(tokens) {
  return tokens
    .filter((t) => t !== PAD_ID)
    .map((t) => WORDS[t] ?? "<unk>")
    .join(" ");
}

// Example prompts surfaced in the UI.
export const EXAMPLE_PROMPTS = [
  "a sphere", "a cube", "a torus", "a rocket", "a star", "a snowman",
  "a tree", "a chair", "a pyramid", "an arch", "a tall cylinder",
  "a sphere on top of a cube", "a cone next to a cylinder", "two cubes",
];
