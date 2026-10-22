---
title: La botica - pharma-assistant
emoji: 💊
colorFrom: green
colorTo: yellow
sdk: static
app_file: index.html
pinned: false
license: apache-2.0
short_description: ATC classifier in your browser and real agent receipts
---

# La botica / The pharmacy cabinet

Static demo of [pharma-assistant](https://github.com/hsilvosa/Pharmaceutical-Assistant-and-ATC-Classifier), English by default with a Spanish switch.

- **ATC classifier, in your browser.** BETO fine-tuned on AEMPS CIMA, exported to ONNX and quantised to int8 (110 MB), run with Transformers.js. It predicts only the anatomical group (ATC level 1, 14 groups). Probabilities are temperature-scaled (expected calibration error 0.189 to 0.061).
- **Agent receipts, real runs.** Nine recorded runs of the retrieval-augmented agent: steps, timings, refusals and verbatim source quotes.

Information from official leaflets, not medical advice. Data: AEMPS CIMA. No data leaves your browser.
