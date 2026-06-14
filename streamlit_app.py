from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st
import torch
from torch import nn
from torchvision import datasets


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from oxpets.config import DATA_DIR, FIGURES_DIR, METRICS_DIR, TrainConfig, select_device, set_seed
from oxpets.data import BREED_CLASSES, SPECIES_CLASSES, make_loaders
from oxpets.models import build_scratch_model
from oxpets.train import evaluate, train_one_epoch, trainable_parameters


st.set_page_config(page_title="Oxford Pets Project", layout="wide")


@st.cache_resource
def load_raw_dataset():
    return datasets.OxfordIIITPet(
        root=DATA_DIR,
        split="trainval",
        target_types="category",
        transform=None,
        download=True,
    )


def show_image_grid() -> None:
    dataset = load_raw_dataset()
    indices = [0, 280, 560, 840, 1120, 1400, 1680, 1960]
    cols = st.columns(4)
    for item_id, index in enumerate(indices):
        image, label = dataset[index]
        with cols[item_id % 4]:
            st.image(image, caption=BREED_CLASSES[int(label)], use_container_width=True)


def show_saved_results() -> None:
    summary_path = METRICS_DIR / "model_comparison_summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        st.dataframe(summary, use_container_width=True, hide_index=True)
    else:
        st.info("Run python main.py to create the final metrics table.")

    figures = [
        ("Model comparison", FIGURES_DIR / "model_comparison_summary.png"),
        ("Species predictions", FIGURES_DIR / "species_transfer_prediction_grid.png"),
        ("Species training curves", FIGURES_DIR / "species_transfer_history.png"),
        ("Breed confusion matrix", FIGURES_DIR / "breed_transfer_confusion_matrix.png"),
    ]

    for title, path in figures:
        if path.exists():
            st.subheader(title)
            st.image(str(path), use_container_width=True)


def unnormalize(image_tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (image_tensor.cpu() * std + mean).clamp(0, 1)


def show_predictions(model, loader, device: torch.device) -> None:
    model.eval()
    images, labels = next(iter(loader))
    with torch.no_grad():
        logits = model(images.to(device))
        probabilities = torch.softmax(logits, dim=1).cpu()
        predictions = probabilities.argmax(dim=1)
        confidence = probabilities.max(dim=1).values

    st.subheader("Prediction examples")
    cols = st.columns(4)
    for index in range(min(8, len(images))):
        true_label = SPECIES_CLASSES[int(labels[index])]
        predicted_label = SPECIES_CLASSES[int(predictions[index])]
        score = float(confidence[index])
        caption = f"true: {true_label} | pred: {predicted_label} ({score:.2f})"
        with cols[index % 4]:
            image = unnormalize(images[index]).permute(1, 2, 0).numpy()
            st.image(image, caption=caption, use_container_width=True)


def run_live_training(epochs: int, train_limit: int, image_size: int) -> None:
    set_seed(42)
    device = select_device()
    config = TrainConfig(image_size=image_size, batch_size=16, num_workers=0)

    train_loader, val_loader, test_loader, spec = make_loaders(
        task="species",
        config=config,
        download=True,
        limit_train=train_limit,
        limit_val=max(64, train_limit // 4),
        limit_test=256,
    )

    model = build_scratch_model(spec.num_classes).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(trainable_parameters(model), lr=0.001, weight_decay=0.0001)

    progress = st.progress(0)
    status = st.empty()
    chart_area = st.empty()
    table_area = st.empty()

    history: list[dict] = []
    for epoch in range(1, epochs + 1):
        status.write(f"Training epoch {epoch} of {epochs} on {device}")
        train_loss, train_accuracy = train_one_epoch(model, train_loader, loss_function, optimizer, device)
        val_loss, val_accuracy, _, _ = evaluate(model, val_loader, loss_function, device)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        })

        frame = pd.DataFrame(history).set_index("epoch")
        chart_area.line_chart(frame[["train_accuracy", "val_accuracy"]])
        table_area.dataframe(pd.DataFrame(history), use_container_width=True, hide_index=True)
        progress.progress(epoch / epochs)

    test_loss, test_accuracy, y_true, y_pred = evaluate(model, test_loader, loss_function, device)
    status.success(f"Finished. Test accuracy: {test_accuracy:.3f}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Test accuracy", f"{test_accuracy:.3f}")
    col2.metric("Test loss", f"{test_loss:.3f}")
    col3.metric("Test images", len(y_true))

    show_predictions(model, test_loader, device)


def main() -> None:
    st.title("Oxford Pets Image Classification")
    st.write("Live project demo for the Deep Learning and Big Data final presentation.")

    tab_overview, tab_data, tab_results, tab_live = st.tabs([
        "Overview",
        "Dataset",
        "Results",
        "Live Training",
    ])

    with tab_overview:
        st.header("Project idea")
        st.write(
            "We classify pet images from the Oxford-IIIT Pet dataset. "
            "The project compares a CNN trained from scratch with a ResNet18 transfer learning model."
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("Breed classes", "37")
        col2.metric("Species classes", "2")
        col3.metric("Models compared", "2")

        st.subheader("Main tasks")
        st.write("- Breed classification: 37 classes")
        st.write("- Species classification: cat vs dog")

    with tab_data:
        st.header("Dataset examples")
        st.write("These are real images from the Oxford-IIIT Pet dataset.")
        show_image_grid()

    with tab_results:
        st.header("Saved project results")
        st.write("These figures were generated from the project code and used in the report and slides.")
        show_saved_results()

    with tab_live:
        st.header("Live training")
        st.write("This trains the scratch CNN on the cat vs dog task and updates the chart after every epoch.")

        col1, col2, col3 = st.columns(3)
        epochs = col1.slider("Epochs", min_value=2, max_value=8, value=4)
        train_limit = col2.slider("Training images", min_value=256, max_value=1024, value=512, step=128)
        image_size = col3.selectbox("Image size", [96, 128, 160], index=1)

        if st.button("Start live training", type="primary"):
            run_live_training(epochs=epochs, train_limit=train_limit, image_size=image_size)


if __name__ == "__main__":
    main()
