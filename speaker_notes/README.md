# Speaker Notes Overview

These notes are for a 15 minute presentation. Each student should understand the whole project, but each person can lead the part connected to their contribution.

## Suggested Timing

| Time | Speaker | Main focus |
|---|---|---|
| 0:00-3:30 | Love | project goal, dataset, official Oxford Pets facts, EDA, preprocessing |
| 3:30-6:30 | Jamal Dassrath | scratch CNN, training loop, loss, optimizer, scratch results |
| 6:30-10:00 | Erbakan Ahmad | transfer learning, ResNet18, frozen head, fine-tuning, comparison |
| 10:00-13:30 | Muhammad Ahtisham Bhatti | results story, report and slides, limitations, conclusion |
| 13:30-15:00 | All | Streamlit demo, final message, Q&A handoff |

## Slide Flow

1. Start with the assignment requirement and our two tasks.
2. Show the Oxford Pets dataset and explain why it is useful for image classification.
3. Explain preprocessing, train/validation/test split, and reproducibility.
4. Explain the scratch CNN as the course-based model.
5. Explain the ResNet18 transfer model as the stronger practical model.
6. Show final metrics, training curves, confusion matrices, and prediction examples.
7. Finish with limitations, next steps, and the Streamlit live demo.

## Handoff Cues

- Love to Jamal: "Now that the dataset and preprocessing are clear, Jamal will explain our first model, the CNN trained from scratch."
- Jamal to Erbakan: "The scratch CNN shows the course concepts well, but the results also show why pretrained features help. Erbakan will explain our transfer learning model."
- Erbakan to Muhammad: "After training both approaches, we compared them in the report and slides. Muhammad will explain the results and conclusion."
- Muhammad to team demo: "To finish, we will show the Streamlit app so the professor can see the dataset, figures, and live training behavior."

## Q&A Preparation

- If asked why we used two tasks: the assignment asks for two tasks on one dataset, so we used breed classification and cat-vs-dog classification.
- If asked why breed classification is harder: it has 37 classes and some breeds look visually similar.
- If asked why transfer learning works better: ResNet18 already learned useful image features from ImageNet.
- If asked whether the final evaluation is fair: the official test split is kept separate and used only after training.
- If asked whether Streamlit live training uses the full data: no, it uses a smaller demo subset for speed. The report results use the full official test split.
