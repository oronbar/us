# Cardiotoxicity presentation sources

These scripts were preserved from temporary authoring folders alongside the
three final PowerPoint presentations at the repository root.

- `feature_importance/build_feature_deck.mjs` builds the original importance
  deck. Keep its slide modules, runtime, and content-tokens.json together.
- `feature_families/edit_feature_family_slide.mjs` edits template-starter.pptx
  to produce the feature-family version. The template differs from the final
  original deck and is retained as the actual editing input.
- `models_explained/build_deck.mjs` builds the detailed model explanation.

Run with Node.js in an environment providing `@oai/artifact-tool`. Dependencies
are not vendored. Original scripts use D:/us paths for inputs and final outputs;
adjust those paths for another checkout. Preview and layout files are ignored.

Cleanup validation checks script syntax only. The existing decks were preserved
without regeneration.
