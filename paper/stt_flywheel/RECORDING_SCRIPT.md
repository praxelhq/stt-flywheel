# Pushpak's Te recording script (n=20, ~5 min audio)
**Goal:** human-recorded sanity check for the paper. Each sentence below is one row from our entity-dense Te eval set; you read it, we transcribe with all 4 systems, we check whether β-Te's EHR holds on real human speech (vs the synth-only number we currently report).
## Setup
- **Device:** phone voice memos app or laptop mic. No studio needed.
- **Room:** quiet — close door, no fan, away from a/c.
- **Mic distance:** ~6-12 inches from your mouth, consistent across all 20.
- **Files:** save each as `pushpak_te_NN.wav` (or `.m4a` — we'll convert) where NN = `01`..`20`. Drop them all in one folder named `pushpak_te_recordings/`.
- **Style:** read each sentence naturally, like you'd say it on a phone call. Don't over-articulate. If you stumble, restart that one sentence (no editing needed; just re-record that take).
- **Pause between recordings:** stop & start the recorder per sentence. Keeps file naming clean.

**Class distribution (n=20):**

| Class | Count | What it tests |
|---|---|---|
| brands | 4 | English brand names embedded in Te (Vodafone, Flipkart, etc.) |
| addresses | 4 | Indian-style with embedded plot numbers / pincodes |
| currency | 3 | Amounts in Te words and Latin numerals |
| codemix | 4 | English carrier verbs + Te content nouns or vice versa |
| digits | 3 | Digit strings (phone numbers, OTPs) |
| proper_nouns | 2 | Indian person/place names |

Total recording time should be ~5-8 minutes including pauses. **Send me the folder when done** — I'll run the 4-system eval and add a row to Table 1 of the paper showing whether β-Te's entity-dense gain holds on your real audio.

---

## 01. [brands] entities: `Paytm`

> నా స్నేహితుడు Paytm లో నిర్వాహకుడిగా ఉన్నాడు

## 02. [brands] entities: `Paytm`

> Paytm గ్లోబల్ ట్రాన్సఫర్ సేవ చాలా వేగవంతమైనది

## 03. [brands] entities: `Paytm`

> నేను ప్రతిదిన Paytm ఉపయోగిస్తాను

## 04. [brands] entities: `Dunzo`

> డెలివరీ కోసం Dunzo వాడతాను నేను

## 05. [addresses] entities: `234`

> నిజామాబాద్ నగరంలో ప్లాట్ సంఖ్య 234, సదరంగంజ్ కాలనీలో

## 06. [addresses] entities: `123`

> పూణెలో 123-A, సిటీ సెంటర్‌లో ఆ భవనం ఉంది

## 07. [addresses] entities: `3`, `678`

> కోచిన్‌ 682018లో 3-678, మట్టంచెర్రి వద్ద నివసిస్తున్నాడు

## 08. [addresses] entities: `110`

> రంగారెడ్డిపేట రోడ్ మీద ఉన్న ఇంటి చిరునామా 110 అవతార్ నగర్ హైదరాబాద్

## 09. [currency] entities: (no tagged tokens)

> నిలువుగా జీఎస్టీ ఇరవై లక్ష రూపాయల ఆధారంపై

## 10. [currency] entities: `22,000`

> నెల ఆరంభానికి రూ. 22,000 అద్దె చెల్లించాలి

## 11. [currency] entities: `2`, `3`

> ఆ నగరం ఆదాయం 2 కోటల నుండి 3 కోటల

## 12. [codemix] entities: `Network`, `connectivity`, `issue`, `hai`, `IT`, `team`, `contact`, `karo`

> Network connectivity issue hai, IT team contact karo.

## 13. [codemix] entities: `training`, `schedule`

> సార్, నీ training schedule ఈ నెలలో ఉంటుందట

## 14. [codemix] entities: `sir`, `priority`, `high`, `notification`

> sir, priority high ఉందని notification వచ్చింది

## 15. [codemix] entities: `Repository`, `push`, `commit`, `message`, `clear`

> Repository push చేసా, commit message clear ఉందా?

## 16. [digits] entities: (no tagged tokens)

> నా రిఫరెన్స్ నంబర్ ఆరు ఏడు ఎనిమిది నాలుగు రెండు

## 17. [digits] entities: (no tagged tokens)

> జీవో నంబర్ ఒకటి ఒకటి ఎనిమిది ఏడు రెండు తొమ్మిది

## 18. [digits] entities: (no tagged tokens)

> ఆ కోడ్ ఏడు ఏడు ఏడు ఆరు ఆరు ఆరు ఐదు

## 19. [proper_nouns] entities: (no tagged tokens)

> ఢిల్లీ లో చాలా ఠంఢా ఉంది

## 20. [proper_nouns] entities: (no tagged tokens)

> కొచ్చిన్ నుండి సుగంధ దార్వాలు తీసుకోవాలి

---

**When done:** zip the folder + send via the same file-share you've been using. I'll handle the rest:
1. Convert m4a→wav 16kHz mono if needed.
2. Build `data/stt_flywheel/holdouts/te/entity_dense_pushpak_native.jsonl` with the same id-text-entity_tokens schema as the synth set.
3. Run vanilla v3 + Te-r2 + vasista22 + β-Te + Deepgram on it.
4. Add the result as Table 6 in the paper: 'Native human-recorded sanity check.'
5. If β-Te EHR on your recordings ≥ 0.30, that's a strong defense against the synth-train/synth-test reviewer concern. If ≥ 0.40, headline-defensible.
