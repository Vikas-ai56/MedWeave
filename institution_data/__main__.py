"""`python -m institution_data prepare`: provision the institutional PathMNIST shards and print the proof.

Downloads the pinned file if needed, builds one shard per institution, then re-reads every shard and
prints each institution's sample count and per-class counts, the overall class proportions, and whether
the shards are disjoint, cover every official training index once, and match the manifest. Exits 1 if
any check fails. Options: --data-dir (default $MEDWEAVE_DATA_DIR or ~/.medweave/data), --seed (default
0), --institutions (default inst-1 inst-2 inst-3 inst-4).
"""

import argparse
import sys
from pathlib import Path

from institution_data.pathmnist import CLASS_NAMES
from institution_data.provision import DEFAULT_INSTITUTION_IDS, audit_partitions, default_data_dir, prepare_pathmnist


def print_report(data_dir: Path, seed: int) -> bool:
    audit = audit_partitions(data_dir, seed)
    manifest = audit.manifest
    print(f"{manifest.dataset} partition seed {manifest.seed}, source MD5 {manifest.source_md5}")
    print(f"official training samples: {manifest.total_samples}")
    header = "class".ljust(38) + "".join(entry.institution_id.rjust(9) for entry in manifest.institutions)
    print(header + "   share")
    totals = [sum(entry.class_counts[label] for entry in manifest.institutions) for label in range(manifest.num_classes)]
    for label, name in enumerate(CLASS_NAMES[: manifest.num_classes]):
        counts = "".join(str(entry.class_counts[label]).rjust(9) for entry in manifest.institutions)
        print(f"{label} {name}".ljust(38) + counts + f"   {totals[label] / manifest.total_samples:6.2%}")
    print("samples".ljust(38) + "".join(str(entry.num_samples).rjust(9) for entry in manifest.institutions))
    print(f"disjoint: {audit.disjoint}   every index once: {audit.complete}   shards match manifest: {audit.shards_match_manifest}")
    return audit.passed


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m institution_data")
    parser.add_argument("command", choices=["prepare"])
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--institutions", nargs="+", default=list(DEFAULT_INSTITUTION_IDS))
    arguments = parser.parse_args()
    data_dir = arguments.data_dir or default_data_dir()
    prepare_pathmnist(data_dir, arguments.institutions, arguments.seed)
    return 0 if print_report(data_dir, arguments.seed) else 1


if __name__ == "__main__":
    sys.exit(main())
