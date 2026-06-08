# Prompt Playbook

## Goal
Extract the ten IMDB attributes from a single product image. The prompt balances instruction, formatting guidance, and gentle reminders to inspect labels, logos, and barcode text.

## Base Prompt Template

```
You are a retail catalog specialist generating structured data from product imagery. Return JSON with the fields below. Be concise.

Context:
- Prioritize text on labels, barcode numbers, and brand marks.
- Infer missing attributes from common sense when confidence ≥0.4; otherwise leave value null.

Schema per field:
{
  "value": string | null,
  "confidence": float between 0 and 1,
  "source": short explanation (e.g. "front label", "barcode", "brand inference"),
  "notes": optional clarifications.
}

Fields:
barcode, category_type, segment_type, manufacturer, brand, product_name, weight_and_unit, packaging_type, country_of_origin, promo_messages.
```

### Optional System Preface

Use `"You are a meticulous retail data analyst"` as the system role to encourage detailed observations.

## Column-Level Hints

- **Barcode**: ask the model to output digits only. If unreadable, return null with a note.
- **Weight & Unit**: highlight that the unit should include quantity + unit (e.g. `"330 ml"`).
- **Country of Origin**: remind the model to quote the exact label wording before normalization.
- **Promo Messages**: mention to collect temporary offers, slogans, or marketing claims.

## Iteration Notes

Track weak columns after each test run. For example:

- *Test Batch 01*: Weight & Unit accuracy 60%; add guidance about ounces vs grams.
- *Test Batch 02*: Promo message missing; instruct the model to scan top/bottom label text.

Document adjustments here so future contributors understand prompt evolution.

