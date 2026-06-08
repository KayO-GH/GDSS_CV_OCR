# Prompt Playbook

## Goal
Extract the 13 hackathon IMDB fields from all images in a product group. The prompt balances instruction, formatting guidance, and reminders to inspect labels, logos, barcode text, manufacturer text, and the image tag printed at the bottom of sample images.

## Base Prompt Template

```
You are a retail catalog specialist generating structured data from grouped product imagery. Return JSON with the fields below. Be concise.

Context:
- Prioritize text on labels, barcode numbers, and brand marks.
- Combine evidence across all product angles.
- Return uppercase values.
- Use digit-only barcodes and compact weights such as `250G`, `500ML`, `1L`, or `2.2KG`.
- Leave uncertain values null instead of guessing.

Schema per field:
{
  "value": string | null,
  "confidence": float between 0 and 1,
  "source": short explanation (e.g. "front label", "barcode", "brand inference"),
  "notes": optional clarifications.
}

Fields:
item_name, barcode, manufacturer, brand, weight, packaging_type, country, variant, type, fragrance_flavor, promotion, addons, tagline.
```

### Optional System Preface

Use `"You are a meticulous retail data analyst"` as the system role to encourage detailed observations.

## Column-Level Hints

- **Barcode**: ask the model to output digits only. If unreadable, return null with a note.
- **Item Name**: use the full descriptive catalog name, including brand, weight, packaging, type, manufacturer, and country when visible.
- **Weight**: compact quantity + unit with no internal space unless the ground truth style requires it.
- **Country of Origin**: remind the model to quote the exact label wording before normalization.
- **Promotion/Addons/Tagline**: split temporary offers, pack contents, and descriptive slogans into the correct fields.

## Iteration Notes

Track weak columns after each test run. For example:

- *Test Batch 01*: Weight & Unit accuracy 60%; add guidance about ounces vs grams.
- *Test Batch 02*: Promo message missing; instruct the model to scan top/bottom label text.

Document adjustments here so future contributors understand prompt evolution.
