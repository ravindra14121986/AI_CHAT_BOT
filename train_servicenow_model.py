"""
train_servicenow_model.py
==========================
End-to-end trainer for a spaCy pipeline (NER + intent classification / textcat)
on the ServiceNow incident chatbot dataset, using an NVIDIA GPU via CUDA.

WHAT THIS SCRIPT DOES
----------------------
1. Verifies the GPU / cupy / spaCy setup and prints a clear diagnostic.
2. Loads servicenow_incident_training_data.json (text, {"intent", "entities"}).
3. Splits into train/dev.
4. Converts each example into a spaCy Doc carrying BOTH:
     - doc.ents      (from "entities")   -> trains the `ner` component
     - doc.cats      (from "intent")     -> trains the `textcat` component
   and serializes them into ./corpus/train.spacy and ./corpus/dev.spacy
5. Auto-generates a spaCy v3 training config (tok2vec shared by ner + textcat).
6. Runs `spacy train` on the GPU (or CPU if you pass --gpu-id -1).
7. Loads the best resulting model and runs a few sanity-check predictions.

USAGE
-----
    python train_servicenow_model.py \\
        --data /mnt/user-data/outputs/servicenow_incident_training_data.json \\
        --gpu-id 0

Run with --gpu-id -1 to force CPU training (useful for a quick smoke test
before committing to a full GPU run).

REQUIREMENTS (install BEFORE running this script)
--------------------------------------------------
    # 1) Core packages
    pip install -U spacy

    # 2) GPU array library matching your CUDA 13.1 install.
    #    Do NOT use `pip install spacy[cudaXXX]` - those extras pin old
    #    cupy versions that conflict with current numpy/thinc. Install
    #    cupy directly instead:
    pip install cupy-cuda13x

    #    If you don't already have the CUDA 13.x toolkit installed system-wide,
    #    you can instead pull the needed runtime libraries via pip with:
    pip install "cupy-cuda13x[ctk]"

See INSTALL.md (included alongside this script) for the full walkthrough
and troubleshooting notes.
"""

import argparse
import json
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent


# ----------------------------------------------------------------------
# 0. GPU / ENVIRONMENT CHECK
# ----------------------------------------------------------------------

def check_environment(gpu_id: int) -> bool:
    """Validate spaCy, CuPy, CUDA, and GPU availability.

    Args:
        gpu_id: CUDA device index. Use ``-1`` to skip GPU checks and force CPU mode.

    Returns:
        ``True`` when GPU training is enabled and spaCy accepts the requested GPU;
        otherwise ``False`` for CPU mode.

    Raises:
        SystemExit: If spaCy or CuPy is missing, or if GPU initialization fails.

    Example:
        ``check_environment(0)`` checks the first CUDA GPU, while
        ``check_environment(-1)`` selects CPU mode.
    """
    print("=" * 70)
    print("ENVIRONMENT CHECK")
    print("=" * 70)

    try:
        import spacy
        print(f"spaCy version: {spacy.__version__}")
    except ImportError:
        print("ERROR: spaCy is not installed. Run: pip install -U spacy")
        sys.exit(1)

    if gpu_id < 0:
        print("GPU disabled by --gpu-id -1. Training will run on CPU.")
        return False

    try:
        import cupy
        n_devices = cupy.cuda.runtime.getDeviceCount()
        print(f"cupy version: {cupy.__version__}")
        print(f"CUDA devices detected: {n_devices}")
        if n_devices == 0:
            print("WARNING: cupy is installed but no CUDA device was found.")
            return False
        props = cupy.cuda.runtime.getDeviceProperties(gpu_id)
        name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
        print(f"Using device {gpu_id}: {name}")
        runtime_version = cupy.cuda.runtime.runtimeGetVersion()
        print(f"CUDA runtime version (as reported by cupy): {runtime_version}")
    except ImportError:
        print(
            "ERROR: cupy is not installed, so spaCy cannot use the GPU.\n"
            "        Install it with:  pip install cupy-cuda13x\n"
            "        (or pip install 'cupy-cuda13x[ctk]' if you don't have\n"
            "        the CUDA 13.x toolkit installed system-wide)."
        )
        sys.exit(1)
    except Exception as e:  # pragma: no cover - diagnostic path
        print(f"WARNING: could not fully query the GPU ({e}). Continuing anyway.")

    import spacy
    ok = spacy.require_gpu(gpu_id)
    print(f"spacy.require_gpu({gpu_id}) -> {ok}")
    return True


# ----------------------------------------------------------------------
# 1. DATA LOADING / CONVERSION TO .spacy BINARY FORMAT
# ----------------------------------------------------------------------

def load_data(path: Path):
    """Load the JSON training dataset from disk.

    The expected format is a list of ``[text, annotations]`` pairs, where each
    annotation contains an ``intent`` string and an ``entities`` list.

    Args:
        path: Path to ``servicenow_incident_training_data.json``.

    Returns:
        The decoded Python list containing all training examples.

    Example:
        A valid JSON item looks like::

            ["What is INC1234567?", {
                "intent": "get_incident_by_number",
                "entities": [[8, 18, "INCIDENT_NUMBER"]]
            }]
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def build_docbins(data, dev_frac: float, seed: int):
    """Convert raw examples into annotated training and development DocBins.

    Each generated spaCy document receives entity annotations for NER and a
    one-hot ``doc.cats`` dictionary for exclusive intent classification. The
    examples are shuffled deterministically before the train/dev split.

    Args:
        data: List of ``[text, annotations]`` training examples.
        dev_frac: Fraction of examples reserved for validation.
        seed: Random seed used for reproducible shuffling.

    Returns:
        A tuple containing the train DocBin, dev DocBin, and sorted intent labels.

    Example:
        ``build_docbins(data, dev_frac=0.1, seed=42)`` reserves 10 percent of
        the examples for validation and produces repeatable output.
    """
    import spacy
    from spacy.tokens import DocBin

    random.seed(seed)
    random.shuffle(data)

    all_intents = sorted({ann["intent"] for _, ann in data})
    print(f"Found {len(all_intents)} intent labels: {all_intents}")

    n_dev = int(len(data) * dev_frac)
    dev_data = data[:n_dev]
    train_data = data[n_dev:]
    print(f"Train examples: {len(train_data)} | Dev examples: {len(dev_data)}")

    nlp = spacy.blank("en")

    def make_docbin(examples):
        """Build one DocBin from examples and report unaligned entities.

        Example:
            ``make_docbin(train_data)`` converts the selected training examples
            into serialized spaCy documents.
        """
        db = DocBin()
        dropped_spans = 0
        for text, ann in examples:
            doc = nlp.make_doc(text)

            # --- entities -> ner ---
            ents = []
            for start, end, label in ann["entities"]:
                span = doc.char_span(start, end, label=label, alignment_mode="contract")
                if span is None:
                    dropped_spans += 1
                    continue
                ents.append(span)
            doc.ents = ents

            # --- intent -> textcat (single-label / exclusive) ---
            doc.cats = {intent: 1.0 if intent == ann["intent"] else 0.0 for intent in all_intents}

            db.add(doc)
        if dropped_spans:
            print(f"  (note: {dropped_spans} entity spans could not be aligned to token "
                  f"boundaries and were dropped)")
        return db

    train_db = make_docbin(train_data)
    dev_db = make_docbin(dev_data)
    return train_db, dev_db, all_intents


# ----------------------------------------------------------------------
# 2. CONFIG GENERATION
# ----------------------------------------------------------------------

def generate_config(config_path: Path, gpu: bool):
    """Generate a complete spaCy v3 training config.

    spaCy first creates a base configuration for the NER and textcat pipeline,
    then fills in all architecture defaults so training can run without
    interactive configuration. The generated config is written to
    ``config_path``.

    Args:
        config_path: Destination for the filled training configuration.
        gpu: Whether the config generator should optimize defaults for GPU use.

    Example:
        ``generate_config(Path("run/config.cfg"), gpu=True)`` creates a
        GPU-oriented config at ``run/config.cfg``.
    """
    base_config = config_path.parent / "base_config.cfg"

    cmd = [
        sys.executable, "-m", "spacy", "init", "config",
        str(base_config),
        "--lang", "en",
        "--pipeline", "ner,textcat",
        "--optimize", "efficiency",
        "--force",
    ]
    if gpu:
        cmd.append("--gpu")

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    # `init config` only drafts the file; `init fill-config` resolves every
    # default so `spacy train` doesn't have to guess anything at run time.
    fill_cmd = [
        sys.executable, "-m", "spacy", "init", "fill-config",
        str(base_config), str(config_path),
    ]
    print("Running:", " ".join(fill_cmd))
    subprocess.run(fill_cmd, check=True)


# ----------------------------------------------------------------------
# 3. TRAINING
# ----------------------------------------------------------------------

def run_training(config_path: Path, train_path: Path, dev_path: Path,
                  output_dir: Path, gpu_id: int):
    """Launch spaCy training as a subprocess.

    Args:
        config_path: Filled spaCy training configuration.
        train_path: Serialized training DocBin path.
        dev_path: Serialized validation DocBin path.
        output_dir: Directory where ``model-best`` and ``model-last`` are saved.
        gpu_id: CUDA device index, or ``-1`` for CPU training.

    Raises:
        subprocess.CalledProcessError: If spaCy training exits unsuccessfully.

    Example:
        ``run_training(Path("run/config.cfg"), Path("train.spacy"),
        Path("dev.spacy"), Path("run/output"), gpu_id=0)`` trains on GPU 0.
    """
    cmd = [
        sys.executable, "-m", "spacy", "train",
        str(config_path),
        "--output", str(output_dir),
        "--paths.train", str(train_path),
        "--paths.dev", str(dev_path),
        "--gpu-id", str(gpu_id),
    ]
    print("=" * 70)
    print("TRAINING")
    print("=" * 70)
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


# ----------------------------------------------------------------------
# 4. SANITY-CHECK INFERENCE
# ----------------------------------------------------------------------

def sanity_check(model_dir: Path):
    """Load the best model and print predictions for representative requests.

    This is a smoke test after training. It reports the highest-scoring intent
    and extracted entities for several ServiceNow-style user queries.

    Args:
        model_dir: Path to the trained ``model-best`` directory.

    Example:
        ``sanity_check(Path("servicenow_model_run/output/model-best"))``
        loads the model and prints predictions for sample requests.
    """
    import spacy

    print("=" * 70)
    print("SANITY CHECK")
    print("=" * 70)
    nlp = spacy.load(model_dir)

    samples = [
        "Create a new incident, the VPN keeps dropping, priority 2 - High.",
        "What is the status of INC1234567?",
        "Assign INC7654321 to Sarah Johnson.",
        "Close INC1112223 with resolution code Solved (Permanently).",
        "Show me all incidents assigned to Network Team.",
    ]
    for text in samples:
        doc = nlp(text)
        top_intent = max(doc.cats.items(), key=lambda kv: kv[1]) if doc.cats else ("N/A", 0)
        print(f"\nTEXT: {text}")
        print(f"  Predicted intent: {top_intent[0]}  (score={top_intent[1]:.3f})")
        print(f"  Entities: {[(ent.text, ent.label_) for ent in doc.ents]}")


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

def main():
    """Parse command-line options and run the complete training workflow.

    The workflow checks the environment, loads JSON data, creates train/dev
    DocBins, generates a config, trains the model, and performs a sanity check
    on the resulting best model.

    Example:
        Run the complete workflow on the first GPU with::

            python train_servicenow_model.py --gpu-id 0

        For a CPU smoke test, use ``--gpu-id -1``.
    """
    parser = argparse.ArgumentParser(description="Train ServiceNow incident NER + intent model on GPU")
    parser.add_argument("--data", type=Path,
                         default=PROJECT_DIR / "servicenow_incident_training_data.json",
                         help="Path to the training data JSON.")
    parser.add_argument("--work-dir", type=Path, default=PROJECT_DIR / "servicenow_model_run",
                         help="Directory to hold corpus/, config, and trained model output.")
    parser.add_argument("--gpu-id", type=int, default=0,
                         help="CUDA device id to use. Pass -1 to train on CPU.")
    parser.add_argument("--dev-split", type=float, default=0.1,
                         help="Fraction of data held out for the dev/validation set.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    use_gpu = check_environment(args.gpu_id)

    args.work_dir.mkdir(parents=True, exist_ok=True)
    corpus_dir = args.work_dir / "corpus"
    corpus_dir.mkdir(exist_ok=True)
    config_path = args.work_dir / "config.cfg"
    output_dir = args.work_dir / "output"

    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    data = load_data(args.data)
    print(f"Loaded {len(data)} examples from {args.data}")
    intent_counts = Counter(ann["intent"] for _, ann in data)
    for intent, count in intent_counts.most_common():
        print(f"  {intent:35s} {count}")

    train_db, dev_db, all_intents = build_docbins(data, args.dev_split, args.seed)
    train_path = corpus_dir / "train.spacy"
    dev_path = corpus_dir / "dev.spacy"
    train_db.to_disk(train_path)
    dev_db.to_disk(dev_path)
    print(f"Wrote {train_path} and {dev_path}")

    print("=" * 70)
    print("GENERATING CONFIG")
    print("=" * 70)
    generate_config(config_path, gpu=use_gpu)

    run_training(
        config_path=config_path,
        train_path=train_path,
        dev_path=dev_path,
        output_dir=output_dir,
        gpu_id=args.gpu_id if use_gpu else -1,
    )

    best_model_dir = output_dir / "model-best"
    if best_model_dir.exists():
        sanity_check(best_model_dir)
        print("\n" + "=" * 70)
        print(f"DONE. Best model saved at: {best_model_dir}")
        print("=" * 70)
    else:
        print("WARNING: model-best was not found - check the training logs above for errors.")


if __name__ == "__main__":
    main()
