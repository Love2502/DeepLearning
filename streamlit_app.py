from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import torch
from torch import nn
from PIL import Image
from torchvision import datasets


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from oxpets.config import DATA_DIR, FIGURES_DIR, METRICS_DIR, TrainConfig, select_device, set_seed
from oxpets.data import BREED_CLASSES, SPECIES_CLASSES, make_loaders
from oxpets.models import build_scratch_model, build_transfer_model
from oxpets.train import evaluate, train_one_epoch, trainable_parameters


st.set_page_config(page_title="Oxford Pets Project", layout="wide")


@st.cache_resource
def load_raw_dataset(split: str = "trainval"):
    return datasets.OxfordIIITPet(
        root=DATA_DIR,
        split=split,
        target_types="category",
        transform=None,
        download=True,
    )


@st.cache_data(show_spinner="Reading Oxford Pets image metadata...")
def load_dataset_metadata() -> pd.DataFrame:
    rows: list[dict] = []
    for split in ["trainval", "test"]:
        dataset = datasets.OxfordIIITPet(
            root=DATA_DIR,
            split=split,
            target_types="category",
            transform=None,
            download=True,
        )
        for path, breed_label, species_label in zip(dataset._images, dataset._labels, dataset._bin_labels):
            with Image.open(path) as image:
                width, height = image.size
            rows.append({
                "split": split,
                "path": str(path),
                "breed": BREED_CLASSES[int(breed_label)],
                "species": SPECIES_CLASSES[int(species_label)],
                "width": width,
                "height": height,
                "megapixels": round((width * height) / 1_000_000, 3),
            })
    return pd.DataFrame(rows)


def show_image_grid() -> None:
    dataset = load_raw_dataset("trainval")
    indices = [0, 280, 560, 840, 1120, 1400]
    cols = st.columns(2)
    for item_id, index in enumerate(indices):
        image, label = dataset[index]
        caption = f"{BREED_CLASSES[int(label)]} | original image {image.width} x {image.height}"
        with cols[item_id % 2]:
            st.image(image, caption=caption, width="stretch")


def draw_bar_chart(frame: pd.DataFrame, x_column: str, y_column: str, title: str, color: str) -> None:
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.bar(frame[x_column], frame[y_column], color=color)
    ax.set_title(title)
    ax.set_ylabel("Images")
    ax.tick_params(axis="x", rotation=90)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def draw_image_size_plot(metadata: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].scatter(metadata["width"], metadata["height"], s=9, alpha=0.35, color="#2563eb")
    axes[0].set_title("Original image width and height")
    axes[0].set_xlabel("Width")
    axes[0].set_ylabel("Height")
    axes[0].grid(alpha=0.25)

    axes[1].hist(metadata["megapixels"], bins=24, color="#16a34a", alpha=0.85)
    axes[1].set_title("Original image-size distribution")
    axes[1].set_xlabel("Megapixels")
    axes[1].set_ylabel("Images")
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def show_dataset_exploration() -> None:
    metadata = load_dataset_metadata()
    trainval_count = int((metadata["split"] == "trainval").sum())
    test_count = int((metadata["split"] == "test").sum())

    st.subheader("Official Dataset Facts")
    st.write(
        "The official Oxford page describes this as a 37-category pet dataset with roughly "
        "200 images per class. It also notes large variation in scale, pose, and lighting. "
        "The official annotations include species and breed name, head ROI, and trimap segmentation."
    )
    st.caption("Official source: https://www.robots.ox.ac.uk/~vgg/data/pets/")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Trainval images", trainval_count)
    col2.metric("Test images", test_count)
    col3.metric("Breed classes", len(BREED_CLASSES))
    col4.metric("Species classes", len(SPECIES_CLASSES))

    st.subheader("Label Distribution")
    trainval = metadata[metadata["split"] == "trainval"]
    breed_counts = trainval["breed"].value_counts().reindex(BREED_CLASSES).reset_index()
    breed_counts.columns = ["breed", "images"]
    species_counts = trainval["species"].value_counts().reindex(SPECIES_CLASSES).reset_index()
    species_counts.columns = ["species", "images"]

    left, right = st.columns([2, 1])
    with left:
        draw_bar_chart(breed_counts, "breed", "images", "Breed labels in trainval split", "#3b82f6")
    with right:
        draw_bar_chart(species_counts, "species", "images", "Cat vs dog labels", "#f97316")

    st.subheader("Original Image Sizes")
    draw_image_size_plot(metadata)

    st.subheader("Explore Real Images")
    controls = st.columns(3)
    selected_split = controls[0].selectbox("Split", ["trainval", "test"])
    selected_species = controls[1].selectbox("Species", ["all", *SPECIES_CLASSES])
    selected_breed = controls[2].selectbox("Breed", ["all", *BREED_CLASSES])

    gallery = metadata[metadata["split"] == selected_split]
    if selected_species != "all":
        gallery = gallery[gallery["species"] == selected_species]
    if selected_breed != "all":
        gallery = gallery[gallery["breed"] == selected_breed]

    if gallery.empty:
        st.warning("No images match this filter.")
        return

    gallery = gallery.sample(n=min(8, len(gallery)), random_state=42).reset_index(drop=True)
    cols = st.columns(4)
    for index, row in gallery.iterrows():
        image = Image.open(row["path"]).copy()
        caption = f"{row['breed']} | {row['species']} | {row['width']} x {row['height']}"
        with cols[index % 4]:
            st.image(image, caption=caption, width="stretch")


def show_saved_results() -> None:
    summary_path = METRICS_DIR / "model_comparison_summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        st.dataframe(summary, width="stretch", hide_index=True)
    else:
        st.info("Run python main.py to create the final metrics table.")

    figures = [
        ("Dataset examples", FIGURES_DIR / "dataset_sample_grid.png"),
        ("Dataset label distribution", FIGURES_DIR / "dataset_label_distribution.png"),
        ("Model comparison", FIGURES_DIR / "model_comparison_summary.png"),
        ("Breed predictions", FIGURES_DIR / "breed_transfer_prediction_grid.png"),
        ("Breed training curves", FIGURES_DIR / "breed_transfer_history.png"),
        ("Breed confusion matrix", FIGURES_DIR / "breed_transfer_confusion_matrix.png"),
        ("Species predictions", FIGURES_DIR / "species_transfer_prediction_grid.png"),
        ("Species training curves", FIGURES_DIR / "species_transfer_history.png"),
        ("Species confusion matrix", FIGURES_DIR / "species_transfer_confusion_matrix.png"),
    ]

    for title, path in figures:
        if path.exists():
            st.subheader(title)
            image = Image.open(path).copy()
            modified = pd.Timestamp(path.stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M")
            st.caption(f"{image.width} x {image.height} result image, updated {modified}")
            st.image(image, width="stretch")


def unnormalize(image_tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (image_tensor.cpu() * std + mean).clamp(0, 1)


def tensor_to_display_image(image_tensor: torch.Tensor, size: int = 520) -> Image.Image:
    image = unnormalize(image_tensor).permute(1, 2, 0).numpy()
    image = Image.fromarray((image * 255).astype("uint8"))
    return image.resize((size, size), Image.Resampling.LANCZOS)


def show_predictions(model, loader, device: torch.device, class_names: list[str]) -> None:
    model.eval()
    images, labels = next(iter(loader))
    with torch.no_grad():
        logits = model(images.to(device))
        probabilities = torch.softmax(logits, dim=1).cpu()
        predictions = probabilities.argmax(dim=1)
        confidence = probabilities.max(dim=1).values

    st.subheader("Prediction examples")
    cols = st.columns(2)
    for index in range(min(8, len(images))):
        true_label = class_names[int(labels[index])]
        predicted_label = class_names[int(predictions[index])]
        score = float(confidence[index])
        caption = f"true: {true_label} | pred: {predicted_label} ({score:.2f})"
        with cols[index % 2]:
            image = tensor_to_display_image(images[index])
            st.image(image, caption=caption, width="stretch")


def run_live_training(task: str, model_name: str, epochs: int, train_limit: int, image_size: int) -> None:
    set_seed(23)
    device = select_device()
    config = TrainConfig(image_size=image_size, batch_size=16, num_workers=0)

    train_loader, val_loader, test_loader, spec = make_loaders(
        task=task,
        config=config,
        download=True,
        limit_train=train_limit,
        limit_val=max(64, train_limit // 4),
        limit_test=max(256, train_limit // 2),
    )

    if model_name == "Transfer learning":
        model = build_transfer_model(spec.num_classes, freeze_backbone=True, pretrained=True).to(device)
    else:
        model = build_scratch_model(spec.num_classes).to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(trainable_parameters(model), lr=0.001, weight_decay=0.0001)

    progress = st.progress(0)
    status = st.empty()
    chart_area = st.empty()
    table_area = st.empty()

    history: list[dict] = []
    for epoch in range(1, epochs + 1):
        status.write(f"Training {task} {model_name.lower()} epoch {epoch} of {epochs} on {device}")
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
        table_area.dataframe(pd.DataFrame(history), width="stretch", hide_index=True)
        progress.progress(epoch / epochs)

    test_loss, test_accuracy, y_true, y_pred = evaluate(model, test_loader, loss_function, device)
    status.success(f"Finished. Test accuracy: {test_accuracy:.3f}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Test accuracy", f"{test_accuracy:.3f}")
    col2.metric("Test loss", f"{test_loss:.3f}")
    col3.metric("Test images", len(y_true))

    show_predictions(model, test_loader, device, spec.class_names)


def show_final_task_gallery(task: str) -> None:
    if task == "breed":
        figures = [
            ("Final breed predictions from the trained transfer model", FIGURES_DIR / "breed_transfer_prediction_grid.png"),
            ("Breed training curves", FIGURES_DIR / "breed_transfer_history.png"),
            ("Breed confusion matrix", FIGURES_DIR / "breed_transfer_confusion_matrix.png"),
        ]
    else:
        figures = [
            ("Final species predictions from the trained transfer model", FIGURES_DIR / "species_transfer_prediction_grid.png"),
            ("Species training curves", FIGURES_DIR / "species_transfer_history.png"),
            ("Species confusion matrix", FIGURES_DIR / "species_transfer_confusion_matrix.png"),
        ]

    for title, path in figures:
        if path.exists():
            st.subheader(title)
            image = Image.open(path).copy()
            st.image(image, width="stretch")


def main() -> None:
    st.title("Oxford Pets Image Classification")
    st.write("Live project demo for the Deep Learning and Big Data final presentation.")

    tab_overview, tab_data, tab_results, tab_live = st.tabs([
        "Overview",
        "Data Exploration",
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
        st.header("Data Exploration")
        st.write("These plots and images are generated from the local Oxford Pets dataset files.")
        show_dataset_exploration()

    with tab_results:
        st.header("Saved project results")
        st.write("These figures were generated from the project code and used in the report and slides.")
        show_saved_results()

    with tab_live:
        st.header("Live training")
        st.write("This trains a model live and updates the chart after every epoch.")
        st.warning(
            "Live training uses a smaller demo subset so it can finish during the presentation. "
            "The final report results use the full official test split."
        )

        col1, col2, col3, col4 = st.columns(4)
        task_label = col1.selectbox("Task", ["Breed classification", "Cat vs dog classification"])
        model_name = col2.selectbox("Model", ["Transfer learning", "Scratch CNN"])
        epochs = col3.slider("Epochs", min_value=1, max_value=8, value=3)
        image_size = col4.selectbox("Image size", [128, 160, 224], index=1)

        task = "breed" if task_label == "Breed classification" else "species"
        default_train = 1024 if task == "breed" else 512
        train_limit = st.slider(
            "Training images for the live demo",
            min_value=256,
            max_value=2048,
            value=default_train,
            step=128,
        )

        if task == "breed":
            st.info(
                "Breed classification has 37 classes, so transfer learning is the best live-demo choice. "
                "The scratch CNN is still available, but it needs longer training to look strong."
            )
        else:
            st.info("Cat vs dog is the easier task, so both models can learn visible patterns quickly.")

        if st.button("Start live training", type="primary"):
            run_live_training(
                task=task,
                model_name=model_name,
                epochs=epochs,
                train_limit=train_limit,
                image_size=image_size,
            )

        st.divider()
        st.header("Final trained results for this task")
        show_final_task_gallery(task)


if __name__ == "__main__":
    main()
