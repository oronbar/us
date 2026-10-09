# Research update for the meeting with Dan

The final seven-slide Hebrew presentation is in
`deliverables/meeting_with_dan.pptx`, with a PDF viewing copy alongside it.
Charts and narrative text in the PowerPoint remain editable. Speaker notes
record the source files, cohort definitions, and interpretation limits.

The deck covers temporal sampling, same-visit GLS prediction, vendor differences,
next-visit video fusion, EchoPrime adaptation and view selection, curve feature
importance, and proposed next steps. It distinguishes the non-CNN feature
importance study from the video benchmarks and from the separate landmark study.

## Build sources

- `.build/prepare_data.py`: extracts anonymous prediction pairs and aggregate
  feature importance from the existing local research outputs.
- `.build/data.json`: the anonymous numeric inputs used for this deck.
- `.build/build.mjs`: creates editable slides with the bundled Artifact Tool and
  validates the PPTX. Paths refer to the original Windows authoring environment.
- `.build/fix_rtl.py`: applies Hebrew paragraph direction to the exported OOXML.

Building requires the bundled Node/Artifact Tool runtime and Python with lxml.
The Node runtime package directory must be available through the build directory's
`node_modules` link and the `RUNTIME_NODE_MODULES` environment variable. The
finalizer deliberately refuses to overwrite an existing output, so choose a new
output filename for a revision. The PDF was exported using Microsoft PowerPoint.

Temporary renders, logs, dependency links, and superseded drafts remain local.
