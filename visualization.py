# @title Organize the output and display a sample of the predictions

rows = []

for embeddings, label in validation_data.batch(1):
  row = {
      f'{DIAGNOSIS}_prediction': model(embeddings)[DIAGNOSIS].numpy().flatten()[0],
      f'{DIAGNOSIS}_value': label.numpy().flatten()[0]
  }
  rows.append(row)

eval_df = pd.DataFrame(rows)
eval_df.head()


import sklearn
import matplotlib.pyplot as plt

def plot_curve(x, y, auc, x_label=None, y_label=None, label=None):
  fig = plt.figure(figsize=(10, 10))
  plt.plot(x, y, label=f'{label} (AUC: %.3f)' % auc, color='black')
  plt.legend(loc='lower right', fontsize=18)
  plt.xlim([-0.01, 1.01])
  plt.ylim([-0.01, 1.01])
  if x_label:
    plt.xlabel(x_label, fontsize=24)
  if y_label:
    plt.ylabel(y_label, fontsize=24)
  plt.xticks(fontsize=12)
  plt.yticks(fontsize=12)
  plt.grid(visible=True)

#%matplotlib inline
labels = eval_df[f'{DIAGNOSIS}_value'].values
predictions = eval_df[f'{DIAGNOSIS}_prediction'].values
false_positive_rate, true_positive_rate, thresholds = sklearn.metrics.roc_curve(
    labels,
    predictions,
    drop_intermediate=False)
auc = sklearn.metrics.roc_auc_score(labels, predictions)
plot_curve(false_positive_rate, true_positive_rate, auc, x_label='False Positive Rate', y_label='True Positive Rate', label=DIAGNOSIS)


# @title Show Sample Images (prediction scores vs label)

import numpy as np
from PIL import Image, ImageOps, ImageDraw
import matplotlib.pyplot as plt
from matplotlib import gridspec
import io
import pandas as pd
from google.colab import output

NUM_BUCKETS=5

thumbnails = np.load(THUMBNAILS_FILE_PATH, allow_pickle=True)

output.no_vertical_scroll()

sorted_indices = np.argsort(predictions)
sorted_predictions = predictions[sorted_indices]
sorted_labels = labels[sorted_indices]
sorted_image_id = df_validate["image_id"].values[sorted_indices]
sorted_image_id, sorted_labels, sorted_predictions


def define_buckets(predictions, num_buckets):
    quantiles = pd.Series(predictions).quantile([i / num_buckets for i in range(num_buckets + 1)]).tolist()
    # Make sure the range covers the last value inclusively
    quantiles[-1] = quantiles[-1] + 0.01
    return [(quantiles[i], quantiles[i + 1]) for i in range(len(quantiles) - 1)]

def bucket_images(sorted_predictions, sorted_labels, num_buckets=4):
    # Define buckets
    buckets = define_buckets(sorted_predictions, num_buckets)
    bucketed_images = {bucket: {0: [], 1: []} for bucket in buckets}

    # Loop over all predictions, labels, and images to organize them into the buckets
    for index, (score, label) in enumerate(zip(sorted_predictions, sorted_labels)):
        for bucket in buckets:
            if bucket[0] <= score < bucket[1]:
                bucketed_images[bucket][label].append(index)  # Store the index instead of the image
                break
    return bucketed_images


def plot_bucketed_images(bucketed_images, sorted_predictions, sorted_image_id, thumbnails):
    num_columns = 2  # (2 for Label 0 and Label 1)
    num_rows = len(bucketed_images)

    desired_height = 300  # height in pixels for each image
    desired_width = 300   # width in pixels for each image

    # Create the figure with specified size, considering the new image dimensions
    fig = plt.figure(figsize=(20, (num_rows + 1) * (desired_height / 100)), constrained_layout=True)
    gs = gridspec.GridSpec(nrows=num_rows + 1, ncols=num_columns + 1, figure=fig,
                           width_ratios=[1, 5, 5], height_ratios=[1] + [5] * num_rows)
    fig.patch.set_facecolor('#ffe9d2')

    # Initialize the axes array using the GridSpec
    axes = np.empty((num_rows + 1, num_columns + 1), dtype=object)
    for i in range(num_rows + 1):
        for j in range(num_columns + 1):
            axes[i, j] = plt.subplot(gs[i, j])
            for spine in axes[i, j].spines.values():
                spine.set_visible(True)
                spine.set_linewidth(2)
                spine.set_edgecolor('black')

    # Setting title rows and column labels
    axes[0, 1].set_title(f"{DIAGNOSIS} Ground Truth - Negative")
    axes[0, 1].axis('off')
    axes[0, 2].set_title(f"{DIAGNOSIS} Ground Truth - Positive")
    axes[0, 2].axis('off')
    axes[0, 0].set_title("Model output score")
    axes[0, 0].axis('off')

    for i, bucket in enumerate(bucketed_images.keys()):
        axes[i + 1, 0].text(0.5, 0.5, f"{bucket[0]:.2f} - {bucket[1]:.2f}",
                            rotation=0, size='large', ha='center', va='center')
        axes[i + 1, 0].axis('off')

    # Plot images in their respective cell, limited to 3 images max
    for i, (bucket, images_dict) in enumerate(bucketed_images.items()):
        for j, label in enumerate(sorted(images_dict.keys())):
            ax = axes[i + 1, j + 1]
            ax.axis('off')
            img_count = len(images_dict[label])
            indices = images_dict[label][:3]  # Limit to the first 3 indices
            combined_image = Image.new('RGB', (desired_width * 3, desired_height), 'grey')
            x_offset = 0
            for idx in indices:
                img = Image.open(io.BytesIO(thumbnails[sorted_image_id[idx]].tobytes()))
                img_resized = img.resize((desired_width, desired_height), Image.Resampling.LANCZOS)
                combined_image.paste(img_resized, (x_offset, 0))
                draw = ImageDraw.Draw(combined_image)
                text = f"ID: {sorted_image_id[idx]}\nScore: {sorted_predictions[idx]:.2f}"
                draw.text((x_offset + 10, 10), text, fill="yellow")
                x_offset += desired_width
            ax.imshow(np.asarray(combined_image), aspect='auto')
            ax.set_title(f"{len(indices)} of {img_count} images")
    plt.show()

print("Display images from the validation set, categorized by the model's prediction score compared to the actual ground truth label. Include up to three images for each category.")

bucketed_images = bucket_images(sorted_predictions, sorted_labels, num_buckets=NUM_BUCKETS)
plot_bucketed_images(bucketed_images, sorted_predictions, sorted_image_id, thumbnails)