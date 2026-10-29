from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
import torch

try:
    import scipy.sparse as sp
except ImportError:
    sp = None


@dataclass
class ModelInputs:
    num_users: int
    num_items: int
    num_nodes: int
    feature_dim: int

    user_mapping: pd.DataFrame
    item_mapping: pd.DataFrame

    train_edges: pd.DataFrame
    val_edges: pd.DataFrame
    test_edges: pd.DataFrame

    train_user_idx: torch.Tensor
    train_item_idx: torch.Tensor
    val_user_idx: torch.Tensor
    val_item_idx: torch.Tensor
    test_user_idx: torch.Tensor
    test_item_idx: torch.Tensor

    edge_index: torch.Tensor
    edge_weight: torch.Tensor
    node_degree: torch.Tensor
    normalized_adj: torch.Tensor

    text_features: torch.Tensor
    image_features: torch.Tensor

    cold_start_item_idx: torch.Tensor

    device: torch.device
    paths: Dict[str, str]
    warnings: List[str]


USER_INDEX_CANDIDATES = ("user_idx", "user_index", "uidx")
USER_ID_CANDIDATES = ("reviewer_id", "user_id", "user")
ITEM_INDEX_CANDIDATES = ("item_idx", "item_index", "iidx")
ITEM_ID_CANDIDATES = ("listing_id", "item_id", "item")


def normalize_id_series(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def first_existing_column(
    frame: pd.DataFrame,
    candidates: Sequence[str],
) -> Optional[str]:
    for column in candidates:
        if column in frame.columns:
            return column
    return None


def find_project_root(
    city: str,
    explicit_root: Optional[Path] = None,
) -> Path:
    if explicit_root is not None:
        root = explicit_root.expanduser().resolve()
        expected = root / "data" / "processed" / city
        if not expected.exists():
            raise FileNotFoundError(
                "The supplied --project-root does not contain:\n"
                f"{expected}"
            )
        return root

    candidates: List[Path] = []
    cwd = Path.cwd().resolve()
    candidates.append(cwd)
    candidates.extend(cwd.parents)

    file_path = Path(__file__).resolve()
    candidates.append(file_path.parent)
    candidates.extend(file_path.parents)

    seen: Set[Path] = set()

    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)

        expected = candidate / "data" / "processed" / city
        if expected.exists():
            return candidate

    raise FileNotFoundError(
        "Could not locate project root automatically.\n"
        "Expected: data/processed/<city>\n"
        "Run from the project or pass --project-root."
    )


def resolve_device(requested: str) -> torch.device:
    requested = requested.strip().lower()

    if requested == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    device = torch.device(requested)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable."
        )

    if device.type == "mps":
        raise RuntimeError(
            "This loader uses sparse graph operations. "
            "Use --device cpu on macOS unless local MPS sparse "
            "support has been verified."
        )

    return device


def safe_torch_load(path: Path) -> Any:
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location="cpu",
        )


def load_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def canonicalize_mapping(
    frame: pd.DataFrame,
    kind: str,
) -> Tuple[pd.DataFrame, str]:
    frame = frame.copy()

    if kind == "user":
        index_candidates = USER_INDEX_CANDIDATES
        id_candidates = USER_ID_CANDIDATES
        canonical_index = "user_idx"
        label = "User mapping"
    elif kind == "item":
        index_candidates = ITEM_INDEX_CANDIDATES
        id_candidates = ITEM_ID_CANDIDATES
        canonical_index = "item_idx"
        label = "Item mapping"
    else:
        raise ValueError(f"Unsupported mapping kind: {kind}")

    index_column = first_existing_column(
        frame, index_candidates
    )
    id_column = first_existing_column(
        frame, id_candidates
    )

    if index_column is None:
        raise ValueError(
            f"{label} has no recognized index column. "
            f"Columns: {list(frame.columns)}"
        )

    if id_column is None:
        raise ValueError(
            f"{label} has no recognized ID column. "
            f"Columns: {list(frame.columns)}"
        )

    if index_column != canonical_index:
        frame.rename(
            columns={index_column: canonical_index},
            inplace=True,
        )

    if (
        frame[canonical_index].isna().any()
        or frame[id_column].isna().any()
    ):
        raise ValueError(
            f"{label} contains missing values."
        )

    frame[canonical_index] = frame[canonical_index].astype(
        np.int64
    )
    frame[id_column] = normalize_id_series(
        frame[id_column]
    )

    if frame[canonical_index].duplicated().any():
        raise ValueError(
            f"{label} contains duplicated {canonical_index}."
        )

    if frame[id_column].duplicated().any():
        raise ValueError(
            f"{label} contains duplicated {id_column}."
        )

    frame = (
        frame
        .sort_values(canonical_index)
        .reset_index(drop=True)
    )

    expected = np.arange(len(frame), dtype=np.int64)
    actual = frame[canonical_index].to_numpy(dtype=np.int64)

    if not np.array_equal(actual, expected):
        raise ValueError(
            f"{label}: {canonical_index} is not contiguous "
            "from 0 to N-1."
        )

    return frame, id_column


def canonicalize_interactions(
    frame: pd.DataFrame,
    split_name: str,
    user_mapping: pd.DataFrame,
    user_id_column: str,
    item_mapping: pd.DataFrame,
    item_id_column: str,
) -> pd.DataFrame:
    frame = frame.copy()

    user_idx_column = first_existing_column(
        frame, USER_INDEX_CANDIDATES
    )
    item_idx_column = first_existing_column(
        frame, ITEM_INDEX_CANDIDATES
    )

    if (
        user_idx_column is not None
        and item_idx_column is not None
    ):
        result = pd.DataFrame({
            "user_idx":
                frame[user_idx_column].astype(np.int64),
            "item_idx":
                frame[item_idx_column].astype(np.int64),
        })
    else:
        raw_user_column = first_existing_column(
            frame, USER_ID_CANDIDATES
        )
        raw_item_column = first_existing_column(
            frame, ITEM_ID_CANDIDATES
        )

        if raw_user_column is None or raw_item_column is None:
            raise ValueError(
                f"{split_name} has no recognizable user/item "
                f"columns. Columns: {list(frame.columns)}"
            )

        temp = frame[
            [raw_user_column, raw_item_column]
        ].copy()

        temp[raw_user_column] = normalize_id_series(
            temp[raw_user_column]
        )
        temp[raw_item_column] = normalize_id_series(
            temp[raw_item_column]
        )

        user_lookup = (
            user_mapping[
                [user_id_column, "user_idx"]
            ]
            .rename(
                columns={
                    user_id_column: raw_user_column
                }
            )
        )

        item_lookup = (
            item_mapping[
                [item_id_column, "item_idx"]
            ]
            .rename(
                columns={
                    item_id_column: raw_item_column
                }
            )
        )

        temp = temp.merge(
            user_lookup,
            on=raw_user_column,
            how="left",
            validate="many_to_one",
        )

        temp = temp.merge(
            item_lookup,
            on=raw_item_column,
            how="left",
            validate="many_to_one",
        )

        missing_users = int(
            temp["user_idx"].isna().sum()
        )
        missing_items = int(
            temp["item_idx"].isna().sum()
        )

        if missing_users > 0 or missing_items > 0:
            raise ValueError(
                f"{split_name} contains IDs absent from mappings. "
                f"missing_users={missing_users}, "
                f"missing_items={missing_items}"
            )

        result = temp[
            ["user_idx", "item_idx"]
        ].astype(np.int64)

    duplicated = int(
        result.duplicated().sum()
    )

    if duplicated > 0:
        raise ValueError(
            f"{split_name} contains {duplicated} duplicated "
            "user-item interactions."
        )

    return result.reset_index(drop=True)


def validate_edge_range(
    frame: pd.DataFrame,
    split_name: str,
    num_users: int,
    num_items: int,
):
    if len(frame) == 0:
        raise ValueError(
            f"{split_name} is empty."
        )

    if not frame["user_idx"].between(
        0, num_users - 1
    ).all():
        raise ValueError(
            f"{split_name} contains out-of-range user_idx."
        )

    if not frame["item_idx"].between(
        0, num_items - 1
    ).all():
        raise ValueError(
            f"{split_name} contains out-of-range item_idx."
        )


def local_edge_set(
    frame: pd.DataFrame,
) -> Set[Tuple[int, int]]:
    return set(
        zip(
            frame["user_idx"].astype(int),
            frame["item_idx"].astype(int),
        )
    )


def build_canonical_train_graph(
    train_edges: pd.DataFrame,
    num_users: int,
    num_items: int,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Build train-only bidirectional graph.

    Global node indexing:
        user node = user_idx
        item node = num_users + item_idx

    Zero-degree item nodes are allowed.
    """

    user_np = train_edges[
        "user_idx"
    ].to_numpy(
        dtype=np.int64,
        copy=True,
    )

    item_global_np = (
        num_users
        +
        train_edges[
            "item_idx"
        ].to_numpy(
            dtype=np.int64,
            copy=True,
        )
    )

    src = np.concatenate(
        [user_np, item_global_np]
    )
    dst = np.concatenate(
        [item_global_np, user_np]
    )

    edge_index_np = np.vstack(
        [src, dst]
    ).astype(
        np.int64,
        copy=False,
    )

    num_nodes = num_users + num_items

    degree_np = np.bincount(
        src,
        minlength=num_nodes,
    ).astype(
        np.int64,
        copy=False,
    )

    src_degree = degree_np[src]
    dst_degree = degree_np[dst]

    if (
        (src_degree <= 0).any()
        or (dst_degree <= 0).any()
    ):
        raise ValueError(
            "Internal degree error while building train graph."
        )

    edge_weight_np = (
        1.0
        /
        np.sqrt(
            src_degree.astype(np.float64)
            *
            dst_degree.astype(np.float64)
        )
    ).astype(np.float32)

    edge_index = torch.from_numpy(
        edge_index_np
    ).long()

    edge_weight = torch.from_numpy(
        edge_weight_np
    ).float()

    node_degree = torch.from_numpy(
        degree_np
    ).long()

    try:
        normalized_adj = (
            torch.sparse_coo_tensor(
                edge_index,
                edge_weight,
                size=(num_nodes, num_nodes),
                dtype=torch.float32,
                check_invariants=True,
            )
            .coalesce()
        )
    except TypeError:
        normalized_adj = (
            torch.sparse_coo_tensor(
                edge_index,
                edge_weight,
                size=(num_nodes, num_nodes),
                dtype=torch.float32,
            )
            .coalesce()
        )

    return (
        edge_index,
        edge_weight,
        node_degree,
        normalized_adj,
    )


def extract_container_value(
    obj: Any,
    keys: Sequence[str],
) -> Any:
    if isinstance(obj, dict):
        for key in keys:
            if key in obj:
                return obj[key]

        if len(obj) == 1:
            return next(iter(obj.values()))

    return obj


def normalize_sparse_tensor(
    obj: Any,
    expected_shape: Tuple[int, int],
) -> torch.Tensor:
    obj = extract_container_value(
        obj,
        (
            "norm_adj",
            "normalized_adj",
            "adj",
            "adjacency",
            "matrix",
        ),
    )

    if sp is not None and sp.issparse(obj):
        coo = obj.tocoo()

        indices = torch.from_numpy(
            np.vstack(
                [coo.row, coo.col]
            ).astype(
                np.int64,
                copy=False,
            )
        ).long()

        values = torch.from_numpy(
            coo.data.astype(
                np.float32,
                copy=False,
            )
        ).float()

        tensor = (
            torch.sparse_coo_tensor(
                indices,
                values,
                size=coo.shape,
                dtype=torch.float32,
            )
            .coalesce()
        )

    elif torch.is_tensor(obj):
        if obj.layout == torch.strided:
            raise ValueError(
                "norm_adj.pt is dense; sparse adjacency expected."
            )

        if obj.layout == torch.sparse_coo:
            tensor = (
                obj
                .detach()
                .cpu()
                .to(dtype=torch.float32)
                .coalesce()
            )
        else:
            tensor = (
                obj
                .detach()
                .cpu()
                .to_sparse_coo()
                .to(dtype=torch.float32)
                .coalesce()
            )

    else:
        raise TypeError(
            "Unsupported norm_adj.pt content type: "
            f"{type(obj)}"
        )

    if tuple(tensor.shape) != expected_shape:
        raise ValueError(
            "norm_adj.pt shape mismatch. "
            f"Found {tuple(tensor.shape)}, "
            f"expected {expected_shape}."
        )

    if not torch.isfinite(
        tensor.values()
    ).all():
        raise ValueError(
            "norm_adj.pt contains NaN/Inf."
        )

    return tensor


def sparse_key_value(
    tensor: torch.Tensor,
) -> Tuple[np.ndarray, np.ndarray]:
    tensor = tensor.coalesce().cpu()

    indices = (
        tensor.indices()
        .numpy()
        .astype(
            np.int64,
            copy=False,
        )
    )

    values = (
        tensor.values()
        .numpy()
        .astype(
            np.float32,
            copy=False,
        )
    )

    num_cols = int(
        tensor.shape[1]
    )

    key = (
        indices[0]
        *
        num_cols
        +
        indices[1]
    )

    order = np.argsort(key)

    return key[order], values[order]


def compare_sparse(
    left: torch.Tensor,
    right: torch.Tensor,
    atol: float = 1e-5,
) -> Tuple[bool, float]:
    left_key, left_value = (
        sparse_key_value(left)
    )
    right_key, right_value = (
        sparse_key_value(right)
    )

    if not np.array_equal(
        left_key,
        right_key,
    ):
        return False, float("inf")

    if len(left_value) == 0:
        return True, 0.0

    max_abs_diff = float(
        np.max(
            np.abs(
                left_value
                -
                right_value
            )
        )
    )

    return (
        max_abs_diff <= atol,
        max_abs_diff,
    )


def normalize_saved_edge_index(
    obj: Any,
) -> Optional[torch.Tensor]:
    obj = extract_container_value(
        obj,
        ("edge_index", "edges"),
    )

    if not torch.is_tensor(obj):
        return None

    tensor = (
        obj
        .detach()
        .cpu()
        .long()
    )

    if tensor.ndim != 2:
        return None

    if tensor.shape[0] == 2:
        return tensor.contiguous()

    if tensor.shape[1] == 2:
        return (
            tensor
            .transpose(0, 1)
            .contiguous()
        )

    return None


def load_feature_matrix(
    path: Path,
    label: str,
    num_items: int,
    warnings: List[str],
) -> np.ndarray:
    array = np.load(path)

    if array.ndim != 2:
        raise ValueError(
            f"{label} must be 2D. Found {array.shape}."
        )

    if array.shape[0] != num_items:
        raise ValueError(
            f"{label} row count does not match item_mapping. "
            f"rows={array.shape[0]}, items={num_items}"
        )

    if not np.isfinite(array).all():
        raise ValueError(
            f"{label} contains NaN/Inf."
        )

    if array.dtype != np.float32:
        warnings.append(
            f"{label} dtype {array.dtype} was converted to float32."
        )

    return np.array(
        array,
        dtype=np.float32,
        copy=True,
    )


def create_item_mapping_signature(
    item_mapping: pd.DataFrame,
    item_id_column: str,
) -> str:
    ordered = (
        item_mapping
        .sort_values("item_idx")
        .reset_index(drop=True)
    )

    payload = "\n".join(
        f"{int(row.item_idx)}:{getattr(row, item_id_column)}"
        for row in (
            ordered[
                ["item_idx", item_id_column]
            ]
            .itertuples(index=False)
        )
    ).encode("utf-8")

    return hashlib.sha256(
        payload
    ).hexdigest()


def load_model_inputs(
    city: str = "bangkok",
    project_root: Optional[Path] = None,
    device: str = "cpu",
    verbose: bool = True,
    save_metadata: bool = True,
) -> ModelInputs:
    def log(message: str = ""):
        if verbose:
            print(message)

    warnings: List[str] = []

    city = city.strip()
    if not city:
        raise ValueError(
            "city cannot be empty."
        )

    root = find_project_root(
        city=city,
        explicit_root=project_root,
    )

    base_dir = (
        root
        / "data"
        / "processed"
        / city
    )

    feature_dir = base_dir / "features"
    graph_dir = base_dir / "graph"
    model_dir = base_dir / "model"

    paths = {
        "item_mapping":
            feature_dir / "item_mapping.parquet",
        "user_mapping":
            feature_dir / "user_mapping.parquet",
        "text_features":
            feature_dir / "text_features.npy",
        "image_features":
            feature_dir / "image_features.npy",
        "train_interactions":
            graph_dir / "train_interactions.parquet",
        "val_interactions":
            graph_dir / "val_interactions.parquet",
        "test_interactions":
            graph_dir / "test_interactions.parquet",
        "interactions_mapped":
            graph_dir / "interactions_mapped.parquet",
        "edge_index":
            graph_dir / "edge_index.pt",
        "norm_adj":
            graph_dir / "norm_adj.pt",
        "graph_meta":
            graph_dir / "graph_meta.json",
    }

    required_keys = (
        "item_mapping",
        "user_mapping",
        "text_features",
        "image_features",
        "train_interactions",
        "val_interactions",
        "test_interactions",
        "norm_adj",
    )

    print("=" * 72)
    print("STEP 5.1 - PREPARE MODEL INPUTS")
    print("=" * 72)

    log()
    log("=== PROJECT ===")
    log(f"Project root: {root}")
    log(f"Dataset: {city}")
    log(f"Processed directory: {base_dir}")

    log()
    log("=== REQUIRED FILES ===")

    missing = [
        paths[key]
        for key in required_keys
        if not paths[key].exists()
    ]

    if missing:
        for path in missing:
            log(f"MISSING: {path}")

        raise FileNotFoundError(
            "Required Step 3 / Step 4 files are missing."
        )

    for key in required_keys:
        log(
            "PASS: "
            f"{paths[key].relative_to(root)}"
        )

    log()
    log("=== LOADING MAPPINGS ===")

    user_mapping, user_id_column = (
        canonicalize_mapping(
            pd.read_parquet(
                paths["user_mapping"]
            ),
            "user",
        )
    )

    item_mapping, item_id_column = (
        canonicalize_mapping(
            pd.read_parquet(
                paths["item_mapping"]
            ),
            "item",
        )
    )

    num_users = int(len(user_mapping))
    num_items = int(len(item_mapping))
    num_nodes = num_users + num_items

    log(f"Users: {num_users}")
    log(f"Items: {num_items}")
    log(f"Nodes: {num_nodes}")
    log(f"User ID column: {user_id_column}")
    log(f"Item ID column: {item_id_column}")
    log("Mapping validation: PASS")

    log()
    log("=== LOADING TRAIN / VALIDATION / TEST ===")

    train_edges = canonicalize_interactions(
        pd.read_parquet(
            paths["train_interactions"]
        ),
        "Train split",
        user_mapping,
        user_id_column,
        item_mapping,
        item_id_column,
    )

    val_edges = canonicalize_interactions(
        pd.read_parquet(
            paths["val_interactions"]
        ),
        "Validation split",
        user_mapping,
        user_id_column,
        item_mapping,
        item_id_column,
    )

    test_edges = canonicalize_interactions(
        pd.read_parquet(
            paths["test_interactions"]
        ),
        "Test split",
        user_mapping,
        user_id_column,
        item_mapping,
        item_id_column,
    )

    validate_edge_range(
        train_edges,
        "Train split",
        num_users,
        num_items,
    )

    validate_edge_range(
        val_edges,
        "Validation split",
        num_users,
        num_items,
    )

    validate_edge_range(
        test_edges,
        "Test split",
        num_users,
        num_items,
    )

    train_set = local_edge_set(train_edges)
    val_set = local_edge_set(val_edges)
    test_set = local_edge_set(test_edges)

    if train_set & val_set:
        raise ValueError(
            "Train/validation overlap detected."
        )

    if train_set & test_set:
        raise ValueError(
            "Train/test overlap detected."
        )

    if val_set & test_set:
        raise ValueError(
            "Validation/test overlap detected."
        )

    all_user_idx = set(range(num_users))
    all_item_idx = set(range(num_items))

    train_user_set = set(
        train_edges["user_idx"].astype(int)
    )
    train_item_set = set(
        train_edges["item_idx"].astype(int)
    )

    missing_train_users = sorted(
        all_user_idx - train_user_set
    )

    cold_start_items = sorted(
        all_item_idx - train_item_set
    )

    log(
        f"Train interactions: {len(train_edges)}"
    )
    log(
        f"Validation interactions: {len(val_edges)}"
    )
    log(
        f"Test interactions: {len(test_edges)}"
    )
    log("Split disjointness: PASS")

    if missing_train_users:
        raise ValueError(
            "Train split does not cover every mapped USER. "
            f"Missing users: {len(missing_train_users)}. "
            "This is a cold-start user problem and should be "
            "fixed in Step 4 before model training."
        )

    log("Train user coverage: PASS")

    if cold_start_items:
        warning = (
            f"Train split does not cover {len(cold_start_items)} "
            "mapped item(s). They are retained as cold-start "
            "items because text/image features are available."
        )

        warnings.append(warning)

        log(f"WARNING: {warning}")
        log(
            "Cold-start item_idx preview: "
            f"{cold_start_items[:20]}"
        )
    else:
        log("Train item coverage: PASS")

    if paths["interactions_mapped"].exists():
        mapped_edges = canonicalize_interactions(
            pd.read_parquet(
                paths["interactions_mapped"]
            ),
            "Mapped interactions",
            user_mapping,
            user_id_column,
            item_mapping,
            item_id_column,
        )

        reconstructed = (
            train_set | val_set | test_set
        )

        mapped_set = local_edge_set(
            mapped_edges
        )

        if reconstructed != mapped_set:
            raise ValueError(
                "train + validation + test do not exactly "
                "reconstruct interactions_mapped.parquet."
            )

        log(
            "interactions_mapped reconstruction: PASS"
        )

    log()
    log("=== BUILDING CANONICAL TRAIN GRAPH ===")

    (
        edge_index,
        edge_weight,
        node_degree,
        rebuilt_norm_adj,
    ) = build_canonical_train_graph(
        train_edges,
        num_users,
        num_items,
    )

    user_degrees = node_degree[:num_users]
    item_degrees = node_degree[num_users:]

    zero_degree_users = int(
        (user_degrees == 0).sum().item()
    )
    zero_degree_items = int(
        (item_degrees == 0).sum().item()
    )

    if zero_degree_users != 0:
        raise ValueError(
            f"Canonical graph has {zero_degree_users} "
            "zero-degree user node(s)."
        )

    if zero_degree_items != len(cold_start_items):
        raise ValueError(
            "Cold-start item count does not match "
            "zero-degree item-node count."
        )

    log(
        f"edge_index: {tuple(edge_index.shape)}"
    )
    log(
        f"Directed train graph edges: {edge_index.shape[1]}"
    )
    log(
        f"Zero-degree users: {zero_degree_users}"
    )
    log(
        f"Zero-degree items: {zero_degree_items}"
    )
    log("Canonical graph construction: PASS")

    log()
    log("=== EDGE_INDEX.PT AUDIT ===")

    if paths["edge_index"].exists():
        raw_saved_edge_index = safe_torch_load(
            paths["edge_index"]
        )

        saved_edge_index = normalize_saved_edge_index(
            raw_saved_edge_index
        )

        if saved_edge_index is None:
            warning = (
                "edge_index.pt exists but its format is not "
                "recognized. The canonical model edge_index "
                "rebuilt from train_interactions.parquet will "
                "be used."
            )
            warnings.append(warning)
            log(f"WARNING: {warning}")
        else:
            expected_pairs = set(
                zip(
                    edge_index[0].tolist(),
                    edge_index[1].tolist(),
                )
            )

            saved_pairs = set(
                zip(
                    saved_edge_index[0].tolist(),
                    saved_edge_index[1].tolist(),
                )
            )

            if (
                saved_pairs == expected_pairs
                and saved_edge_index.shape[1]
                == edge_index.shape[1]
            ):
                log(
                    "Saved edge_index matches canonical global "
                    "bidirectional train graph: PASS"
                )
            else:
                warning = (
                    "edge_index.pt uses a different representation "
                    "from the canonical model graph. The canonical "
                    "bidirectional global graph rebuilt from "
                    "train_interactions.parquet will be used."
                )
                warnings.append(warning)
                log(f"WARNING: {warning}")
    else:
        warning = (
            "edge_index.pt not found. Canonical edge_index "
            "was rebuilt from train_interactions.parquet."
        )
        warnings.append(warning)
        log(f"WARNING: {warning}")

    log()
    log("=== NORMALIZED ADJACENCY VALIDATION ===")

    saved_norm_adj = normalize_sparse_tensor(
        safe_torch_load(
            paths["norm_adj"]
        ),
        expected_shape=(
            num_nodes,
            num_nodes,
        ),
    )

    matches, max_abs_diff = compare_sparse(
        saved_norm_adj,
        rebuilt_norm_adj,
        atol=1e-5,
    )

    if not matches:
        raise ValueError(
            "norm_adj.pt does not match the normalized TRAIN "
            "graph rebuilt from train_interactions.parquet. "
            f"Max abs diff: {max_abs_diff}"
        )

    normalized_adj = saved_norm_adj

    log(
        f"norm_adj shape: {tuple(normalized_adj.shape)}"
    )
    log(
        f"norm_adj nnz: {normalized_adj._nnz()}"
    )
    log(
        "Max abs difference vs rebuilt graph: "
        f"{max_abs_diff:.8g}"
    )
    log("Saved normalized adjacency: PASS")

    log()
    log("=== MULTIMODAL FEATURES ===")

    text_np = load_feature_matrix(
        paths["text_features"],
        "Text features",
        num_items,
        warnings,
    )

    image_np = load_feature_matrix(
        paths["image_features"],
        "Image features",
        num_items,
        warnings,
    )

    if text_np.shape != image_np.shape:
        raise ValueError(
            "Text and image feature shapes differ: "
            f"{text_np.shape} vs {image_np.shape}"
        )

    feature_dim = int(
        text_np.shape[1]
    )

    log(
        f"Text features: {text_np.shape}, "
        f"dtype={text_np.dtype}"
    )
    log(
        f"Image features: {image_np.shape}, "
        f"dtype={image_np.dtype}"
    )
    log(
        f"Feature dimension: {feature_dim}"
    )
    log(
        "Feature row alignment with item_mapping: PASS"
    )

    item_signature = create_item_mapping_signature(
        item_mapping,
        item_id_column,
    )

    log(
        f"Item mapping signature: {item_signature}"
    )

    log()
    log("=== CONVERTING TO PYTORCH ===")

    target_device = resolve_device(
        device
    )

    train_user_idx = torch.from_numpy(
        train_edges["user_idx"].to_numpy(
            dtype=np.int64,
            copy=True,
        )
    ).long().to(target_device)

    train_item_idx = torch.from_numpy(
        train_edges["item_idx"].to_numpy(
            dtype=np.int64,
            copy=True,
        )
    ).long().to(target_device)

    val_user_idx = torch.from_numpy(
        val_edges["user_idx"].to_numpy(
            dtype=np.int64,
            copy=True,
        )
    ).long().to(target_device)

    val_item_idx = torch.from_numpy(
        val_edges["item_idx"].to_numpy(
            dtype=np.int64,
            copy=True,
        )
    ).long().to(target_device)

    test_user_idx = torch.from_numpy(
        test_edges["user_idx"].to_numpy(
            dtype=np.int64,
            copy=True,
        )
    ).long().to(target_device)

    test_item_idx = torch.from_numpy(
        test_edges["item_idx"].to_numpy(
            dtype=np.int64,
            copy=True,
        )
    ).long().to(target_device)

    edge_index = edge_index.to(
        target_device
    )
    edge_weight = edge_weight.to(
        target_device
    )
    node_degree = node_degree.to(
        target_device
    )
    normalized_adj = normalized_adj.to(
        target_device
    )

    text_features = torch.from_numpy(
        text_np
    ).float().to(
        target_device
    )

    image_features = torch.from_numpy(
        image_np
    ).float().to(
        target_device
    )

    cold_start_item_idx = torch.tensor(
        cold_start_items,
        dtype=torch.long,
        device=target_device,
    )

    log(
        f"Device: {target_device}"
    )
    log(
        f"torch edge_index: {tuple(edge_index.shape)}"
    )
    log(
        "torch normalized_adj: "
        f"{tuple(normalized_adj.shape)}, "
        f"nnz={normalized_adj._nnz()}"
    )
    log(
        f"torch text_features: {tuple(text_features.shape)}"
    )
    log(
        f"torch image_features: {tuple(image_features.shape)}"
    )
    log(
        f"Cold-start item tensor: "
        f"{tuple(cold_start_item_idx.shape)}"
    )

    log()
    log("=== GRAPH PROPAGATION SMOKE TEST ===")

    probe_dim = 4

    probe = torch.arange(
        num_nodes * probe_dim,
        dtype=torch.float32,
        device=target_device,
    ).reshape(
        num_nodes,
        probe_dim,
    )

    probe = probe / max(
        1,
        num_nodes * probe_dim,
    )

    propagated = torch.sparse.mm(
        normalized_adj,
        probe,
    )

    if tuple(
        propagated.shape
    ) != (
        num_nodes,
        probe_dim,
    ):
        raise ValueError(
            "torch.sparse.mm output shape is incorrect."
        )

    if not torch.isfinite(
        propagated
    ).all():
        raise ValueError(
            "torch.sparse.mm produced NaN/Inf."
        )

    log("torch.sparse.mm: PASS")
    log("Propagation output finite: PASS")

    graph_meta = load_json_if_exists(
        paths["graph_meta"]
    )

    if graph_meta is not None:
        log()
        log("=== GRAPH META ===")
        log(
            "graph_meta.json loaded successfully."
        )

    string_paths = {
        key: str(value)
        for key, value in paths.items()
    }

    result = ModelInputs(
        num_users=num_users,
        num_items=num_items,
        num_nodes=num_nodes,
        feature_dim=feature_dim,

        user_mapping=user_mapping,
        item_mapping=item_mapping,

        train_edges=train_edges,
        val_edges=val_edges,
        test_edges=test_edges,

        train_user_idx=train_user_idx,
        train_item_idx=train_item_idx,
        val_user_idx=val_user_idx,
        val_item_idx=val_item_idx,
        test_user_idx=test_user_idx,
        test_item_idx=test_item_idx,

        edge_index=edge_index,
        edge_weight=edge_weight,
        node_degree=node_degree,
        normalized_adj=normalized_adj,

        text_features=text_features,
        image_features=image_features,

        cold_start_item_idx=(
            cold_start_item_idx
        ),

        device=target_device,
        paths=string_paths,
        warnings=warnings,
    )

    if save_metadata:
        model_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        metadata_path = (
            model_dir
            / "model_input_metadata.json"
        )

        metadata = {
            "created_at":
                datetime.now()
                .astimezone()
                .isoformat(),

            "step":
                "5.1",

            "status":
                "PASS",

            "city":
                city,

            "counts": {
                "users":
                    num_users,
                "items":
                    num_items,
                "nodes":
                    num_nodes,
                "train_interactions":
                    int(len(train_edges)),
                "validation_interactions":
                    int(len(val_edges)),
                "test_interactions":
                    int(len(test_edges)),
                "cold_start_users":
                    0,
                "cold_start_items":
                    int(len(cold_start_items)),
            },

            "graph": {
                "edge_index_shape":
                    list(edge_index.shape),
                "normalized_adj_shape":
                    list(normalized_adj.shape),
                "normalized_adj_nnz":
                    int(normalized_adj._nnz()),
                "zero_degree_users":
                    zero_degree_users,
                "zero_degree_items":
                    zero_degree_items,
                "saved_vs_rebuilt_max_abs_diff":
                    max_abs_diff,
                "train_only":
                    True,
            },

            "features": {
                "text_shape":
                    list(text_np.shape),
                "image_shape":
                    list(image_np.shape),
                "feature_dim":
                    feature_dim,
                "dtype":
                    "float32",
                "item_mapping_signature":
                    item_signature,
            },

            "cold_start_item_idx_preview":
                cold_start_items[:50],

            "device":
                str(target_device),

            "warnings":
                warnings,

            "paths":
                string_paths,
        }

        with open(
            metadata_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                metadata,
                file,
                indent=4,
                ensure_ascii=False,
            )

        log()
        log(
            f"Saved metadata: {metadata_path}"
        )

    log()
    log("=" * 72)
    log("STEP 5.1 COMPLETE")
    log("=" * 72)

    log(f"Users: {num_users}")
    log(f"Items: {num_items}")
    log(f"Nodes: {num_nodes}")
    log(
        "Train / Val / Test: "
        f"{len(train_edges)} / "
        f"{len(val_edges)} / "
        f"{len(test_edges)}"
    )
    log("Cold-start users: 0")
    log(
        f"Cold-start items: {len(cold_start_items)}"
    )
    log(
        f"Directed train graph edges: "
        f"{edge_index.shape[1]}"
    )
    log(
        "Normalized adjacency: "
        f"{tuple(normalized_adj.shape)}, "
        f"nnz={normalized_adj._nnz()}"
    )
    log(
        f"Text features: "
        f"{tuple(text_features.shape)}"
    )
    log(
        f"Image features: "
        f"{tuple(image_features.shape)}"
    )
    log(
        f"Device: {target_device}"
    )

    if warnings:
        log()
        log("WARNINGS:")

        for index, warning in enumerate(
            warnings,
            start=1,
        ):
            log(
                f"{index}. {warning}"
            )

    log()
    log(
        "PASS: Step 5.1 model inputs are validated "
        "and PyTorch-ready."
    )

    if cold_start_items:
        log(
            "NOTE: Cold-start items are retained because "
            "multimodal text/image features are available."
        )

    log("Ready for Step 5.2.")

    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Step 5.1 - prepare model inputs from the current "
            "Step 3 / Step 4 artifact layout."
        )
    )

    parser.add_argument(
        "--city",
        type=str,
        default="bangkok",
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="cpu, auto, cuda, cuda:0",
    )

    parser.add_argument(
        "--no-save-metadata",
        action="store_true",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    load_model_inputs(
        city=args.city,
        project_root=args.project_root,
        device=args.device,
        verbose=True,
        save_metadata=(
            not args.no_save_metadata
        ),
    )


if __name__ == "__main__":
    main()
