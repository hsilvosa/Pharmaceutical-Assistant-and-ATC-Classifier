# What the evaluations found

The numbers in this project are only useful because the evaluations were allowed to fail. These are the places where a first result looked good and was not, in the order they were found. Every figure comes from a file in this repository.

## 1. A 99.9% accuracy that measured nothing

The ATC data has one row per medicine presentation, and about 69% of the texts are duplicates. A first run with a random 85/15 row split reported 99.9% validation accuracy. 97.9% of its validation texts also appeared in training, so the number was memory, not generalisation. It was discarded.

The reported model splits by active-ingredient set, so no ingredient is shared between train, validation and test. On unseen ingredients the top-1 accuracy is 79.3%. Only one seed was run, and class P has no test rows.

## 2. A classifier that was sure and often wrong

On the held-out split the classifier averaged 98.2% confidence at 79.3% accuracy. One temperature (T = 2.37, fitted on validation) lowers the expected calibration error from 0.189 to 0.061 without changing any prediction. The demo shows the calibrated probability.

## 3. Accuracy depends on how the question is typed

The model was trained on a structured text with name, dose, form, ingredients and route. With a name only it reaches 55.1%. With the ingredient only, 68.2%. With name and ingredient, 73.2%. With the full record, 78.8%. The demo asks for the fields separately and prints these figures next to the form.

## 4. Safety rules that looked fine and did not generalise

The first version of the request guard blocked 12 of 39 unsafe requests on its own development set. After a rewrite it blocked all 39, but on 32 new unsafe requests, written after the rules were fixed, it blocked only 13. A model-based intent layer on the same local model blocked 32 of 32 and allowed 30 of 30 safe ones.

Limits: the sets are small, so 32 of 32 supports only a lower bound of about 89% at 95% confidence. One safe held-out question was refused by the rules and fixed afterwards, so the 30 of 30 is not a clean held-out figure.

## 5. A refusal that was a bug

The first free-text run answered 90% of the questions. Six were refused with "select the presentation" because one product appears under several registrations (packs, parallel imports). The resolver now picks the canonical registration for document questions and still asks when the question depends on the registration, for example marketing status or pack.

## 6. A judge that grades its own work

The free-text benchmark has 120 questions written from real leaflet passages. The judge is the same model that generates the answers. A negative control (shuffled answers) was judged correct 1.7% and 0.0% of the time, which shows the judge rejects unrelated answers, not that it is unbiased.

A manual review of the 8 dev answers marked incorrect found 3 judge errors, 3 questions with several valid answers, and 2 real model errors, one of them a renal dose adjustment stated backwards. This is why a pharmacist has to stay in the loop. Test result: 82.3% correct (95% interval 71-90%, n = 62).

## 7. A timeout that was not reproducible

One long leaflet question exceeded the 90 s budget in several benchmark runs. Run alone, it finished in 34 to 43 s. The timeout is intermittent and its cause was not isolated. For that reason the public demo shows it as "slowest run", not as a failure case.

## 8. Language detection by the wrong signal

The agent answered a Spanish question ("¿Qué dosis debo tomar...?") in English. English questions often carry uppercase Spanish medicine names such as CÁPSULAS, so accented letters are not a Spanish signal. The inverted question mark is. The rule now uses it, with tests.

## What this changes in practice

- Report which split a number comes from, and how many items it rests on.
- Fix the evaluation before the model: findings 1, 5 and 8 were data or pipeline problems, not model problems.
- Keep the figures that look bad. They are the part a reader can trust.
