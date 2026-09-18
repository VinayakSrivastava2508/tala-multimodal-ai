# Content-Intent Codebook — Day 2 Task 2 Part F

Defines the weak-label taxonomy in `src/text_features.py::classify_content_intent()`
and `configs/feature_rules.yaml::content_intent_keywords`, for human coders validating
`data/manual_labels/content_intent_validation_sample.csv`.

**This is a weak label.** The rule-based `intent_weak_label` is a keyword-match
starting point for human coding, not a verified classification —
`requires_human_validation=True` on every row. Coders should read the actual
text/media and assign `coder_1_label` / `coder_2_label` independently, then an
adjudicator resolves disagreements into `adjudicated_label`.

## Categories

| Label | Definition | Include | Exclude / ambiguous example |
|-------|------------|---------|------------------------------|
| `awareness` | Introduces the brand/product to an audience that may not know it yet, with no specific call to buy or a named launch. | "Meet TALA — here's who they are and what they stand for." | A caption that also pushes a discount code → likely `conversion` or `mixed` instead. |
| `product_demonstration` | Shows how a specific product looks/fits/performs — try-ons, hauls, unboxings. | "Try-on haul of the new leggings, here's how they fit on me." | If the primary framing is "should you buy this" with a code, lean `conversion`. |
| `product_launch` | Centred on a *new* product, collection, or drop becoming available. | "NEW drop just landed — pre-order now." | A haul of *existing* stock (no "new"/"launching" framing) is `product_demonstration`, not `product_launch`. |
| `conversion` | Explicit purchase-driving language: discount codes, "shop now," "link in bio." | "Use code LUCY10 for 20% off, link in bio." | A caption that only mentions a code once in passing, with the bulk of the content being a review → could be `mixed`. |
| `social_proof` | Testimonial-style content emphasising personal satisfaction/happiness as validation, not a structured review. | "Obsessed with these, best purchase I've made this year." | A structured pros/cons review is `review`, not `social_proof`. |
| `community` | Emphasises belonging, shared identity, "our team," collective language. | "Tag a friend who needs to see this — you're one of us." | Brand-employee content about internal culture → still `community` unless clearly `founder_or_employee` partnership context, which is a *partnership_type* field, not an intent label. |
| `education` | Explains a concept, how-to, or answers a question — not primarily about the product itself. | "How to style leggings for every season." | If the "how-to" is really a thin wrapper around a haul, lean `product_demonstration`. |
| `responsibility` | Centres sustainability/ethics/environmental claims about the brand or product. | "Loving that these are made from recycled ocean plastic." | A single passing mention of "sustainable" inside an otherwise unrelated haul → `mixed` if other signals dominate. |
| `review` | Structured evaluative content — explicit pros/cons, ratings, "honest review," verdict. | "Full honest review: pros, cons, and would I buy again." | A haul with no evaluative judgement (just showing items) is `product_demonstration`. |
| `mixed` | Two or more categories are roughly equally present and neither dominates (the rule-based classifier already applies this when it matches keywords from 2+ categories). | Haul + explicit discount code + sustainability mention, all given similar weight. | If one category clearly dominates despite a passing secondary mention, code the dominant one instead. |
| `unclear` | Text is too short, generic, or off-topic to assign any category confidently. | A caption of just "🩵🩵🩵" or a title with no descriptive content. | If the *media* (not text) clearly signals intent, coders may still assign a label and note the reasoning in `coding_notes`. |

## Coding process

1. Read `text` (title/caption/description) and, where present, view the image at
   `image path or media URL`.
2. Independently assign `coder_1_label` and `coder_2_label` from the table above.
3. Where they disagree, a third pass records `adjudicated_label` with a one-line
   rationale in `coding_notes`.
4. Leave any coder field blank if genuinely undecidable — do not guess to fill the
   cell; use `unclear` and explain in `coding_notes` instead.

## Known limitations

- The rule-based `intent_weak_label` only sees text (title/caption) — it never
  looks at the image, so it can systematically under-detect intent that is purely
  visual (e.g. a studio product shot with a caption that's just an emoji).
- Keyword rules are English-only and TALA/Adanola/Girlfriend Collective/Oner
  Active-context-tuned; they will generalise poorly to other brands or languages.
