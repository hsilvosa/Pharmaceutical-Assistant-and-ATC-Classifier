---
title: La botica - pharma-assistant
emoji: 💊
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: "5.9.1"
app_file: app.py
python_version: "3.11"
pinned: false
license: apache-2.0
short_description: ATC classifier and cited-answer agent over AEMPS CIMA
---

# La botica / The pharmacy cabinet

Interactive demo of [pharma-assistant](https://github.com/hsilvosa/Pharmaceutical-Assistant-and-ATC-Classifier):

- **ATC classifier (live, CPU).** BETO fine-tuned on AEMPS CIMA. It predicts only the anatomical group (ATC level 1, 14 groups) from a Spanish drug description. Probabilities are temperature-scaled; expected calibration error on the held-out split went from 0.189 to 0.061.
- **Agent receipts (static, real runs).** The retrieval-augmented agent needs a GPU, so this tab replays real runs: steps, timings, refusals, and a failure. Answers are rendered only from verbatim quotes of official documents.

Information from official leaflets, not medical advice. Data: AEMPS CIMA. Interface: English (default) and Spanish.
