"""
Cluster visualisation utilities for embedding analysis.

Provides row-based cluster image grids with per-cluster legend panels,
supporting both pure and mixed clusters.
"""

from __future__ import annotations

from pathlib import Path
import textwrap
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import image as mpimg

__all__ = ["ClusterVisualizer"]

warnings.filterwarnings('ignore')
logging.getLogger(__name__).addHandler(logging.NullHandler())

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ClusterVisualizer:
    """Advanced visualiser for clustering results on image embeddings."""

    _PALETTE = {
        'primary_colour': '#2150E0',
        'secondary_colour': ['#4C6DE2', '#90A7F3'],
        'background': '#F5F5F5',
        'line': '#E6E6E6',
        'title': '#171B21',
        'axes': '#313131',
        'subtitle': '#4F4F4F'
    }

    def __init__(self, embeddings: np.ndarray, cluster_labels: np.ndarray,
                 true_labels: np.ndarray, val_samples, label_encoder, seed: int):
        """Initialise the cluster visualiser."""
        if hasattr(embeddings, 'cpu'):
            self.embeddings = embeddings.cpu().numpy().astype(np.float32)
        else:
            self.embeddings = embeddings.astype(np.float32)

        if hasattr(cluster_labels, 'cpu'):
            self.cluster_labels = cluster_labels.cpu().numpy().astype(np.int32)
        else:
            self.cluster_labels = cluster_labels.astype(np.int32)

        if hasattr(true_labels, 'cpu'):
            self.true_labels = true_labels.cpu().numpy().astype(np.int32)
        else:
            self.true_labels = true_labels.astype(np.int32)

        self.val_samples = val_samples
        self.label_encoder = label_encoder
        self.seed = int(seed)
        self.cluster_analysis = self._analyze_cluster_purity()
        self.model_names = self.label_encoder.classes_

        logging.info(
            f"ClusterVisualizer initialised: {len(self.embeddings)} embeddings, "
            f"{len(np.unique(self.cluster_labels))} clusters"
        )

    def _analyze_cluster_purity(self) -> Dict[int, Dict[str, Union[int, float, str]]]:
        """
        Analyse purity statistics for every identified cluster.

        Returns:
            Mapping from cluster ID to per-cluster statistics dict.
        """
        unique_clusters = np.unique(self.cluster_labels)
        cluster_info: Dict[int, Dict[str, Union[int, float, str]]] = {}

        for cluster_id in unique_clusters:
            if cluster_id == -1:
                continue

            mask = self.cluster_labels == cluster_id
            cluster_true_labels = self.true_labels[mask]

            if len(cluster_true_labels) == 0:
                continue

            unique_models = np.unique(cluster_true_labels)
            model_counts = pd.Series(cluster_true_labels).value_counts()

            cluster_size = np.sum(mask)
            most_common_count = int(model_counts.iloc[0])
            purity = float(most_common_count / cluster_size)

            cluster_info[int(cluster_id)] = {
                'size': int(cluster_size),
                'n_unique_models': int(len(unique_models)),
                'most_common_model': int(model_counts.index[0]),
                'most_common_count': most_common_count,
                'purity': purity,
                'diversity_index': 1.0 - purity,
                'entropy': float(-np.sum((model_counts / cluster_size) * 
                                        np.log2(model_counts / cluster_size + 1e-8))),
                'model_distribution': model_counts.to_dict(),
                'is_pure': len(unique_models) == 1,
                'is_mixed': len(unique_models) > 1,
                'is_dominant': purity >= 0.8
            }

        # Map: class -> clusters in which it appears
        class_cluster_map: Dict[int, Dict[str, Any]] = {}
        unique_classes = np.unique(self.true_labels)
        for cls in unique_classes:
            cls_mask = self.true_labels == cls
            clusters_for_class = np.unique(self.cluster_labels[cls_mask])
            clusters_for_class = [int(c) for c in clusters_for_class if int(c) != -1]

            pure_ids = [c for c in clusters_for_class 
                       if cluster_info.get(c, {}).get('is_pure', False)]
            mixed_ids = [c for c in clusters_for_class 
                        if cluster_info.get(c, {}).get('is_mixed', False)]

            class_cluster_map[int(cls)] = {
                'clusters': clusters_for_class,
                'n_clusters': len(clusters_for_class),
                'n_pure_clusters': len(pure_ids),
                'n_mixed_clusters': len(mixed_ids),
                'pure_cluster_ids': pure_ids,
                'mixed_cluster_ids': mixed_ids
            }

        self.class_cluster_map = class_cluster_map

        n_pure = sum(1 for info in cluster_info.values() if info['is_pure'])
        n_mixed = sum(1 for info in cluster_info.values() if info['is_mixed'])
        n_total = len(cluster_info)
        logging.info(f"Clusters analysed: {n_total} ({n_pure} pure, {n_mixed} mixed)")

        multi_cluster_classes = {cls: stats for cls, stats in class_cluster_map.items()
                                 if stats['n_clusters'] > 1}
        if len(multi_cluster_classes) > 0:
            logging.info(f"Classes spread across multiple clusters: {len(multi_cluster_classes)}")

        return cluster_info

    def _get_cluster_legend_text(self, cluster_info: Dict[str, Any], all_indices: List[int], is_pure: bool) -> str:
        """
        Build legend text for a cluster.

        Args:
            cluster_info: Statistics dict for this cluster.
            all_indices: Indices of all samples in the cluster.
            is_pure: Whether the cluster is pure.

        Returns:
            Human-readable legend string.
        """
        if is_pure:
            model_label = cluster_info['most_common_model']
            model_name = self.model_names[model_label].replace('_', ' ')
            count = len(all_indices)
            return f"{model_name} - {count} images"
        else:
            model_dist = cluster_info.get('model_distribution', {})
            if not model_dist:
                return f"Mixed cluster - {len(all_indices)} images"

            model_names = []
            counts = []
            for model_label, count in model_dist.items():
                model_name = self.model_names[model_label].replace('_', ' ')
                model_names.append(model_name)
                counts.append(str(count))

            models_str = ", ".join(model_names)
            counts_str = ", ".join(counts)
            return f"{models_str} - {counts_str} images"

    def _to_numpy_image(self, img) -> Optional[np.ndarray]:
        """
        Convert various image formats to a float32 numpy array in ``[0, 1]``.

        Args:
            img: Image as a tensor, PIL image, or array-like.

        Returns:
            HxWxC numpy array, or ``None`` on failure.
        """
        try:
            if hasattr(img, 'numpy'):
                if len(img.shape) == 3 and img.shape[0] in [1, 3]:
                    img_np = img.permute(1, 2, 0).numpy()
                    if img_np.min() < 0:
                        mean = np.array([0.485, 0.456, 0.406])
                        std = np.array([0.229, 0.224, 0.225])
                        img_np = img_np * std + mean
                    img_np = np.clip(img_np, 0, 1)
                else:
                    img_np = img.numpy()
                    if img_np.max() > 1.0:
                        img_np = img_np / 255.0
            elif hasattr(img, 'convert'):
                img_np = np.array(img.convert('RGB'))
                if img_np.max() > 1.0:
                    img_np = img_np / 255.0
            else:
                img_np = np.array(img)
                if img_np.max() > 1.0:
                    img_np = img_np / 255.0
            
            if len(img_np.shape) == 3 and img_np.shape[0] in [1, 3] and img_np.shape[0] < img_np.shape[1]:
                img_np = np.transpose(img_np, (1, 2, 0))
            
            return img_np
        except Exception as e:
            logging.warning(f"Error converting image: {e}")
            return None

    def _load_image_from_path(self, image_path: str) -> Optional[np.ndarray]:
        """
        Load an image from a file path.

        Args:
            image_path: Absolute or relative path to the image file.

        Returns:
            Numpy array of the image, or ``None`` on failure.
        """
        try:
            img_path = Path(image_path)
            if not img_path.is_absolute():
                repo_relative = PROJECT_ROOT / image_path
                img_path = repo_relative if repo_relative.exists() else Path.cwd() / image_path
            
            img = mpimg.imread(img_path)
            return img
        except Exception as e:
            logging.warning(f"Error loading image from {image_path}: {e}")
            return None

    def _setup_figure_style(self, fig: plt.Figure):
        """
        Apply palette background to the figure.

        Args:
            fig: Matplotlib figure instance.
        """
        fig.patch.set_facecolor(self._PALETTE['background'])

    def _create_grid_axes(self, fig: plt.Figure, n_rows: int, n_cols: int):
        """
        Create a grid of axes with horizontal layout.

        Args:
            fig: Matplotlib figure.
            n_rows: Number of rows.
            n_cols: Number of columns.

        Returns:
            2-D list of ``Axes`` objects.
        """
        gs = plt.GridSpec(n_rows, n_cols, figure=fig,
                         wspace=0.05, hspace=0.25,  
                         left=0.05, right=0.72, top=0.92, bottom=0.05)  
        axes = [[fig.add_subplot(gs[r, c]) for c in range(n_cols)] for r in range(n_rows)]
        for r in range(n_rows):
            for c in range(n_cols):
                axes[r][c].axis('off')  
        return axes

    def _compute_indices_for_pure_cluster(self, cluster_id: int, max_display: int = 5) -> List[int]:
        """
        Select up to *max_display* random sample indices from a pure cluster.

        Args:
            cluster_id: Cluster identifier.
            max_display: Maximum number of images to return.

        Returns:
            List of selected global sample indices.
        """
        cluster_mask = self.cluster_labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]

        k = len(cluster_indices)
        if k == 0:
            return []
        if k <= max_display:
            return [int(i) for i in cluster_indices]

        # Fix seed for reproducibility.
        np.random.seed(self.seed + int(cluster_id))
        sel = np.random.choice(cluster_indices, max_display, replace=False)
        return [int(i) for i in sel]

    def _compute_indices_for_mixed_cluster(self, cluster_id: int) -> List[tuple]:
        """
        Select one random sample per distinct model from a mixed cluster.

        Args:
            cluster_id: Cluster identifier.

        Returns:
            List of ``(global_index, model_label)`` tuples.
        """
        cluster_mask = self.cluster_labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        if len(cluster_indices) == 0:
            return []

        cluster_true_labels = self.true_labels[cluster_mask]
        unique_models = np.unique(cluster_true_labels)

        # One random sample per model
        samples: List[tuple] = []
        for model_label in unique_models:
            local_positions = np.where(cluster_true_labels == model_label)[0]
            if len(local_positions) == 0:
                continue
            np.random.seed(self.seed + int(model_label) + int(cluster_id))
            sel_local = int(np.random.choice(local_positions))
            global_idx = int(cluster_indices[sel_local])
            samples.append((global_idx, int(model_label)))
        return samples

    def _plot_clusters_row(self, clusters: List[Tuple[int, Dict[str, Any]]],
                           indices_per_cluster: List[List[Union[int, tuple]]],
                           all_indices_per_cluster: List[List[int]],
                           fig_title: str,
                           is_pure: bool = False,
                           save_mode: bool = False):
        """
        Render clusters in a row-based layout (one row per cluster).

        Args:
            clusters: List of ``(cluster_id, cluster_info)`` tuples.
            indices_per_cluster: Sample indices to display per cluster.
            all_indices_per_cluster: All indices per cluster (for legend).
            fig_title: Main figure title.
            is_pure: Whether the clusters are pure.
            save_mode: If ``True``, suppress ``plt.show()``.
        """
        if len(clusters) == 0:
            logging.warning("No clusters to display")
            return None

        n_rows = len(clusters)
        n_cols = max(len(lst) for lst in indices_per_cluster) if indices_per_cluster else 0
        if n_cols == 0:
            logging.warning("No images to display")
            return None

        logging.info(f"Rendering {n_rows} clusters with up to {n_cols} images each")

        # Dynamically size figure based on image count
        fig_width = max(14, n_cols * 2.2)
        fig_height = max(8, n_rows * 1.8)
        
        fig = plt.figure(figsize=(fig_width, fig_height), dpi=100)
        fig.patch.set_facecolor(self._PALETTE['background'])  
        axes = self._create_grid_axes(fig, n_rows, n_cols)

        fig.suptitle(fig_title, fontsize=18, fontweight='bold', y=0.97,
                    color=self._PALETTE['title'])

        images_loaded = 0
        total_attempts = 0

        for r, (cid, info) in enumerate(clusters):
            for c, entry in enumerate(indices_per_cluster[r]):
                total_attempts += 1
                try:
                    if isinstance(entry, tuple):
                        global_idx, model_label = entry
                    else:
                        global_idx = int(entry)
                        model_label = None

                    # Attempt to load image from val_samples.
                    try:
                        sample = self.val_samples[global_idx]

                        if isinstance(sample, dict):
                            images = sample.get('images', None)
                        elif hasattr(sample, 'images'):
                            images = sample.images
                        else:
                            # Tuple/list format with paths.
                            if isinstance(sample, (tuple, list)) and len(sample) >= 2:
                                image_paths = sample[1]
                                if isinstance(image_paths, (list, tuple)) and len(image_paths) > 0:
                                    img_np = self._load_image_from_path(image_paths[0])
                                elif isinstance(image_paths, str):
                                    img_np = self._load_image_from_path(image_paths)
                        
                        if img_np is None and 'images' in locals():
                            if images is not None:
                                img = images[0] if isinstance(images, list) else images
                                img_np = self._to_numpy_image(img)
                    except Exception as e:
                        logging.debug(f"val_samples load failed: {e}")

                    # Fall back to error marker
                    if img_np is None:
                        axes[r][c].text(0.5, 0.5, "X", ha="center", va="center", 
                                       fontsize=20, color='red')
                        axes[r][c].set_xlim([0, 1])
                        axes[r][c].set_ylim([0, 1])
                    else:
                        axes[r][c].imshow(img_np)
                        images_loaded += 1

                        for spine in axes[r][c].spines.values():
                            spine.set_visible(True)
                            spine.set_edgecolor(self._PALETTE['line'])
                            spine.set_linewidth(0.5)

                except Exception as e:
                    logging.warning(f"Error displaying image for cluster {cid} position {c}: {e}")
                    axes[r][c].text(0.5, 0.5, "X", ha="center", va="center",
                                   fontsize=20, color='red')

        # Separator lines between cluster rows
        for r in range(n_rows - 1):
            y_pos = 0.92 - ((r + 1) * (0.87 / n_rows))
            line = plt.Line2D([0.05, 0.72], [y_pos, y_pos],
                            transform=fig.transFigure, 
                            color=self._PALETTE['line'], linewidth=1.5, linestyle='--', alpha=0.7, zorder=3)
            fig.add_artist(line)

        # Legend box
        legend_box = plt.Rectangle((0.74, 0.05), 0.24, 0.87,
                                   fill=True, facecolor='white',
                                   edgecolor=self._PALETTE['title'], linewidth=2,
                                   transform=fig.transFigure, zorder=1)
        fig.patches.append(legend_box)

        # Legend title
        title_y = 0.94
        fig.text(0.86, title_y, 'CLUSTER INFORMATION',
                fontsize=11, fontweight='bold', va='center', ha='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=self._PALETTE['secondary_colour'][0],
                          edgecolor=self._PALETTE['title'], linewidth=1.5),
                color=self._PALETTE['title'],
                transform=fig.transFigure,
                zorder=2)

        text_x = 0.755
        max_chars_per_line = 28
        wrapper = textwrap.TextWrapper(width=max_chars_per_line)

        top_image_y = 0.9
        row_height = 0.9 / n_rows

        for r, (cid, info) in enumerate(clusters):
            legend_text = self._get_cluster_legend_text(info, all_indices_per_cluster[r], is_pure)
            y_pos = top_image_y - (r * row_height)

            fig.text(text_x, y_pos, f"Cluster {cid}:",
                    fontsize=9, fontweight='bold', va='top', ha='left',
                    color=self._PALETTE['title'],
                    transform=fig.transFigure,
                    zorder=2)

            wrapped_text = "\n".join(wrapper.wrap(legend_text))
            fig.text(text_x, y_pos - 0.02, wrapped_text,
                    fontsize=8, va='top', ha='left',
                    color=self._PALETTE['axes'],
                    transform=fig.transFigure,
                    zorder=2)
        
        logging.info(f"Render complete: {images_loaded}/{total_attempts} images loaded")
        plt.tight_layout()
        
        if not save_mode:
            plt.show()
        
        return fig

    def visualize_good_clusters(self, n_clusters: int, save_mode: bool = False) -> Optional[plt.Figure]:
        """
        Display pure clusters in a row-based layout.

        Args:
            n_clusters: Number of pure clusters to show.
            save_mode: If ``True``, suppress ``plt.show()``.

        Returns:
            Generated figure, or ``None`` if no pure clusters exist.
        """
        pure_clusters = [(cid, info) for cid, info in self.cluster_analysis.items()
                        if info['is_pure']]
        if len(pure_clusters) == 0:
            logging.warning("No pure clusters found")
            return None

        pure_clusters.sort(key=lambda x: x[1]['size'], reverse=True)
        selected = pure_clusters[:max(1, int(n_clusters))]

        indices_per_cluster: List[List[int]] = []
        all_indices_per_cluster: List[List[int]] = []

        for cid, info in selected:
            display_inds = self._compute_indices_for_pure_cluster(cid, max_display=5)
            all_inds = np.where(self.cluster_labels == cid)[0].tolist()
            indices_per_cluster.append(display_inds)
            all_indices_per_cluster.append(all_inds)

        fig = self._plot_clusters_row(selected, indices_per_cluster, all_indices_per_cluster,
                               fig_title="Pure Clusters",
                               is_pure=True, save_mode=save_mode)
        return fig

    def visualize_mixed_clusters(self, n_clusters: int, max_classes_per_cluster: int = 8, save_mode: bool = False) -> Optional[plt.Figure]:
        """
        Display mixed clusters in a row-based layout.

        Args:
            n_clusters: Number of mixed clusters to show.
            max_classes_per_cluster: Maximum number of classes allowed per cluster.
            save_mode: If ``True``, suppress ``plt.show()``.

        Returns:
            Generated figure, or ``None`` if no viable mixed clusters exist.
        """
        mixed_clusters = [
            (cid, info) for cid, info in self.cluster_analysis.items()
            if info['is_mixed'] and info['n_unique_models'] <= max_classes_per_cluster
        ]

        if len(mixed_clusters) == 0:
            all_mixed = [(cid, info) for cid, info in self.cluster_analysis.items()
                        if info['is_mixed']]
            if len(all_mixed) == 0:
                logging.warning("No mixed clusters found")
            else:
                all_mixed.sort(key=lambda x: x[1]['n_unique_models'], reverse=True)
                max_classes_found = all_mixed[0][1]['n_unique_models'] if all_mixed else 0
                logging.warning(
                    f"No mixed clusters with ≤{max_classes_per_cluster} classes. "
                    f"Smallest cluster has {max_classes_found} classes.")
            return None

        mixed_clusters.sort(key=lambda x: x[1]['n_unique_models'], reverse=True)
        selected = mixed_clusters[:max(1, int(n_clusters))]

        indices_per_cluster: List[List[tuple]] = []
        all_indices_per_cluster: List[List[int]] = []

        for cid, info in selected:
            samples = self._compute_indices_for_mixed_cluster(cid)
            all_inds = np.where(self.cluster_labels == cid)[0].tolist()
            indices_per_cluster.append(samples)
            all_indices_per_cluster.append(all_inds)

        fig = self._plot_clusters_row(selected, indices_per_cluster, all_indices_per_cluster,
                               fig_title="Mixed Clusters",
                               is_pure=False, save_mode=save_mode)
        return fig

    def visualize_best_available_clusters(self, n_clusters: int = 3, max_classes_per_cluster: int = 8) -> None:
        """
        Adaptive strategy: show pure clusters first, then viable mixed ones.

        Args:
            n_clusters: Number of clusters to display.
            max_classes_per_cluster: Maximum classes per cluster for mixed display.
        """
        pure_clusters = [(cid, info) for cid, info in self.cluster_analysis.items()
                        if info['is_pure']]
        mixed_clusters = [(cid, info) for cid, info in self.cluster_analysis.items()
                         if info['is_mixed']]
        viable_mixed = [
            (cid, info) for cid, info in mixed_clusters
            if info['n_unique_models'] <= max_classes_per_cluster
        ]

        logging.info(
            f"Available clusters: {len(pure_clusters)} pure, "
            f"{len(viable_mixed)}/{len(mixed_clusters)} viable mixed"
        )

        if len(pure_clusters) > 0:
            logging.info(f"Showing {min(n_clusters, len(pure_clusters))} pure clusters")
            self.visualize_good_clusters(n_clusters, max_classes_per_cluster)
        elif len(viable_mixed) > 0:
            logging.info(f"Showing {min(n_clusters, len(viable_mixed))} mixed clusters")
            self.visualize_mixed_clusters(n_clusters, max_classes_per_cluster)
        else:
            if len(mixed_clusters) > 0:
                mixed_clusters.sort(key=lambda x: x[1]['n_unique_models'])
                min_classes = mixed_clusters[0][1]['n_unique_models']
                logging.warning(
                    f"Cannot visualise: all clusters have >{max_classes_per_cluster} classes. "
                    f"Minimum found: {min_classes} classes.")
            else:
                logging.warning("No valid clusters to visualise")

    def print_cluster_statistics(self):
        """Log summary statistics for all clusters."""
        total_clusters = len(self.cluster_analysis)
        if total_clusters == 0:
            logging.warning("No clusters to analyse")
            return

        pure_clusters = sum(1 for info in self.cluster_analysis.values() if info['is_pure'])
        dominant_clusters = sum(1 for info in self.cluster_analysis.values() if info['is_dominant'])
        mixed_clusters = sum(1 for info in self.cluster_analysis.values() if info['is_mixed'])

        purities = [info['purity'] for info in self.cluster_analysis.values()]
        entropies = [info['entropy'] for info in self.cluster_analysis.values()]
        sizes = [info['size'] for info in self.cluster_analysis.values()]

        logging.info("=" * 60)
        logging.info("CLUSTERING STATISTICS")
        logging.info("=" * 60)
        logging.info(f"Total clusters: {total_clusters}")
        logging.info(f"Pure clusters: {pure_clusters} ({pure_clusters/total_clusters*100:.1f}%)")
        logging.info(f"Dominant clusters: {dominant_clusters} ({dominant_clusters/total_clusters*100:.1f}%)")
        logging.info(f"Mixed clusters: {mixed_clusters} ({mixed_clusters/total_clusters*100:.1f}%)")
        logging.info(f"Average purity: {np.mean(purities):.3f} ± {np.std(purities):.3f}")
        logging.info(f"Average entropy: {np.mean(entropies):.3f} ± {np.std(entropies):.3f}")
        logging.info(f"Average size: {np.mean(sizes):.1f} ± {np.std(sizes):.1f}")
        logging.info("=" * 60)

    def get_class_cluster_overlap(self) -> pd.DataFrame:
        """
        Return classes that appear in more than one cluster.

        Returns:
            DataFrame with statistics for multi-cluster classes.
        """
        rows = []
        for cls, stats in self.class_cluster_map.items():
            if stats['n_clusters'] <= 1:
                continue
            name = (self.label_encoder.classes_[int(cls)]
                   if hasattr(self.label_encoder, 'classes_') else str(cls))
            rows.append({
                'class_id': int(cls),
                'class_name': name,
                'n_clusters': stats['n_clusters'],
                'n_pure_clusters': stats['n_pure_clusters'],
                'n_mixed_clusters': stats['n_mixed_clusters'],
                'cluster_ids': stats['clusters']
            })
        return pd.DataFrame(rows).sort_values('n_clusters', ascending=False)

    def save_visualizations(
        self,
        output_dir: Union[str, Path],
        n_pure_clusters: int = 15,
        n_mixed_clusters: int = 15,
        n_images_per_cluster: int = 10
    ) -> Dict[str, Path]:
        """
        Save all visualisations to a directory.

        Args:
            output_dir: Directory in which to save the figures.
            n_pure_clusters: Number of pure clusters to visualise.
            n_mixed_clusters: Number of mixed clusters to visualise.
            n_images_per_cluster: Images per cluster.

        Returns:
            Dict mapping figure names to saved file paths.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_files = {}

        try:
            fig = self.visualize_good_clusters(
                n_clusters=n_pure_clusters,
                max_classes_per_cluster=n_images_per_cluster,
                save_mode=True
            )
            if fig is not None:
                pure_path = output_dir / "good_clusters.png"
                fig.savefig(pure_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                saved_files['good_clusters'] = pure_path
                logging.info(f"Pure clusters saved to: {pure_path}")
        except Exception as e:
            logging.warning(f"Could not save pure clusters: {e}")

        try:
            fig = self.visualize_mixed_clusters(
                n_clusters=n_mixed_clusters,
                max_classes_per_cluster=n_images_per_cluster,
                save_mode=True
            )
            if fig is not None:
                mixed_path = output_dir / "mixed_clusters.png"
                fig.savefig(mixed_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                saved_files['mixed_clusters'] = mixed_path
                logging.info(f"Mixed clusters saved to: {mixed_path}")
        except Exception as e:
            logging.warning(f"Could not save mixed clusters: {e}")

        return saved_files