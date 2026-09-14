import argparse
import gzip
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

FASTA_URL = (
    "https://s3.ap-northeast-1.wasabisys.com/gigadb-datasets/live/pub/"
    "10.5524/102001_103000/102523/TgrandC1074.fa.gz"
)
GFF3_URL = (
    "https://s3.ap-northeast-1.wasabisys.com/gigadb-datasets/live/pub/"
    "10.5524/102001_103000/102523/TgrandC1074-Annotation-Primary_Transcripts.gff3.gz"
)

SPLITS = {
    "train": ["chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7"],
    "val": ["chr8"],
    "test": ["chr9", "chr10"],
}

COMPLEMENT = str.maketrans("ACGT", "TGCA")
VALID_BASES = set("ACGT")


def download_if_missing(url, path):
    """Baixa o arquivo em `path` a partir de `url`, se ainda não existir."""
    path = Path(path)
    if path.exists():
        return
    print(f"Baixando {path.name}...", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, path)
    if path.read_bytes()[:2] != b"\x1f\x8b":
        path.unlink()
        sys.exit(f"ERRO: download de {url} não trouxe um .gz válido.")


def load_genome(path):
    """Lê o FASTA e devolve {cromossomo: sequência em maiúsculas}."""
    sequences = {}
    name = None
    chunks = []
    with gzip.open(path, "rt") as handle:
        for line in handle:
            if line.startswith(">"):
                if name is not None:
                    sequences[name] = "".join(chunks).upper()
                name = line[1:].strip().split()[0]
                chunks = []
            else:
                chunks.append(line.strip())
    if name is not None:
        sequences[name] = "".join(chunks).upper()
    return sequences


def load_annotation(path):
    """Lê o GFF3 e devolve (íntrons, genes) como listas de (cromossomo, início, fim, fita)."""
    introns = []
    genes = []
    with gzip.open(path, "rt") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 8:
                continue
            chrom, feature, start, end, strand = (
                parts[0],
                parts[2],
                int(parts[3]),
                int(parts[4]),
                parts[6],
            )
            if feature == "intron":
                introns.append((chrom, start, end, strand))
            elif feature == "gene":
                genes.append((chrom, start, end, strand))
    return introns, genes


def strand_sequence(forward, strand):
    """Sequência do cromossomo na orientação de leitura da fita."""
    if strand == "+":
        return forward
    return forward.translate(COMPLEMENT)[::-1]


def to_strand_position(forward_index, chrom_length, strand):
    """Converte um índice da fita '+' para o índice na sequência já orientada."""
    if strand == "+":
        return forward_index
    return chrom_length - 1 - forward_index


def donor_forward_index(start, end, strand):
    """Índice, na fita '+', da primeira base do íntron: `start` na fita '+', `end` na '-'."""
    return (start - 1) if strand == "+" else (end - 1)


def find_gt_positions(sequence):
    """Índices de todas as ocorrências de 'GT' (posição do 'G')."""
    codes = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
    is_gt = (codes[:-1] == ord("G")) & (codes[1:] == ord("T"))
    return np.flatnonzero(is_gt)


def collect_candidates(sequence, chrom_length, strand, introns, genes):
    """Separa os GT desta fita em positivos, negativos de corpo de gene e negativos do genoma todo."""
    gt_positions = find_gt_positions(sequence)

    is_donor = np.zeros(chrom_length, dtype=bool)
    for _, start, end, _ in introns:
        forward = donor_forward_index(start, end, strand)
        is_donor[to_strand_position(forward, chrom_length, strand)] = True

    # GFF3 é 1-based; o -1 abaixo converte para índice 0-based do Python.
    inside_gene = np.zeros(chrom_length, dtype=bool)
    for _, start, end, _ in genes:
        first = to_strand_position(start - 1, chrom_length, strand)
        last = to_strand_position(end - 1, chrom_length, strand)
        low, high = (first, last) if first <= last else (last, first)
        inside_gene[low : high + 1] = True

    donor_flags = is_donor[gt_positions]
    positives = gt_positions[donor_flags]
    negatives_genome = gt_positions[~donor_flags]
    negatives_gene = gt_positions[inside_gene[gt_positions] & ~donor_flags]

    return positives, negatives_gene, negatives_genome


def extract_windows(sequence, positions, upstream, downstream):
    """Recorta uma janela por posição; descarta se sair da borda ou tiver base fora de ACGT."""
    windows = []
    dropped_edge = 0
    dropped_base = 0
    limit = len(sequence)
    for position in positions:
        start = position - upstream
        end = position + downstream
        if start < 0 or end > limit:
            dropped_edge += 1
            continue
        window = sequence[start:end]
        if not set(window) <= VALID_BASES:
            dropped_base += 1
            continue
        windows.append(window)
    return windows, dropped_edge, dropped_base


def build_dataset(args):
    print(f"Lendo genoma  {args.fasta}", flush=True)
    genome = load_genome(args.fasta)
    print(f"Lendo anotação {args.gff3}", flush=True)
    introns, genes = load_annotation(args.gff3)
    print(
        f"  {len(genome)} cromossomos, "
        f"{len(introns)} íntrons, {len(genes)} genes\n",
        flush=True,
    )

    chrom_to_split = {}
    for split, chroms in SPLITS.items():
        for chrom in chroms:
            chrom_to_split[chrom] = split

    missing = set(chrom_to_split) - set(genome)
    if missing:
        sys.exit(f"ERRO: cromossomos do split ausentes no FASTA: {sorted(missing)}")

    introns_by_key = {}
    genes_by_key = {}
    for record in introns:
        introns_by_key.setdefault((record[0], record[3]), []).append(record)
    for record in genes:
        genes_by_key.setdefault((record[0], record[3]), []).append(record)

    # Localiza os candidatos, sem materializar as sequências ainda.
    candidates = {}
    totals = {split: {"pos": 0, "neg": 0} for split in SPLITS}
    # Estatísticas do genoma inteiro, usadas na seção 1 do notebook.
    genome_stats = {
        "genome_bp": sum(len(seq) for seq in genome.values()),
        "n_chromosomes": len(genome),
        "annotated_introns": len(introns),
        "annotated_genes": len(genes),
        "gt_genome_wide": 0,
        "gt_in_gene_bodies": 0,
        "donor_sites": 0,
    }
    for chrom in sorted(chrom_to_split, key=lambda name: int(name[3:])):
        split = chrom_to_split[chrom]
        length = len(genome[chrom])
        for strand in ("+", "-"):
            sequence = strand_sequence(genome[chrom], strand)
            positives, negatives_gene, negatives_genome = collect_candidates(
                sequence,
                length,
                strand,
                introns_by_key.get((chrom, strand), []),
                genes_by_key.get((chrom, strand), []),
            )
            negatives = (
                negatives_gene if args.neg_pool == "gene-body" else negatives_genome
            )
            candidates[(chrom, strand)] = (positives, negatives)
            totals[split]["pos"] += len(positives)
            totals[split]["neg"] += len(negatives)

            genome_stats["donor_sites"] += len(positives)
            genome_stats["gt_genome_wide"] += len(positives) + len(negatives_genome)
            genome_stats["gt_in_gene_bodies"] += len(positives) + len(negatives_gene)
        print(
            f"  {chrom:6s} [{split:5s}] "
            f"positivos={totals[split]['pos']:7d} (acumulado no split)",
            flush=True,
        )

    print("\nPool natural por split (antes de subamostrar):")
    for split in SPLITS:
        pos = totals[split]["pos"]
        neg = totals[split]["neg"]
        ratio = neg / pos if pos else float("nan")
        print(f"  {split:5s}  positivos={pos:7d}  negativos={neg:9d}  ({ratio:.0f}:1)")

    check_strand_orientation(sum(t["pos"] for t in totals.values()), len(introns))

    # Subamostra depois do split, estratificada por cromossomo.
    rng = np.random.default_rng(args.seed)
    data = {split: {"windows": [], "labels": []} for split in SPLITS}
    drops = {"edge": 0, "base": 0}

    for split, chroms in SPLITS.items():
        # Treino pode ser mais barato; val/teste ficam na razão natural
        # (ver notebook, seção 2).
        evaluating = split in ("val", "test")
        max_pos = args.eval_max_pos if evaluating else args.max_pos
        neg_ratio = args.eval_neg_ratio if evaluating else args.neg_ratio

        pool_pos = totals[split]["pos"]
        pool_neg = totals[split]["neg"]
        target_pos = min(max_pos, pool_pos) if max_pos else pool_pos
        target_neg = min(int(round(target_pos * neg_ratio)), pool_neg)
        if target_neg < round(target_pos * neg_ratio):
            print(
                f"AVISO: {split} tem apenas {pool_neg} negativos disponíveis, "
                f"menos que os {round(target_pos * neg_ratio)} pedidos.",
                file=sys.stderr,
            )

        for chrom in chroms:
            length = len(genome[chrom])
            for strand in ("+", "-"):
                positives, negatives = candidates[(chrom, strand)]
                sequence = strand_sequence(genome[chrom], strand)

                for pool, target, label in (
                    (positives, target_pos, 1),
                    (negatives, target_neg, 0),
                ):
                    total = pool_pos if label == 1 else pool_neg
                    if total == 0 or len(pool) == 0:
                        continue
                    quota = int(round(target * len(pool) / total))
                    quota = min(quota, len(pool))
                    if quota <= 0:
                        continue
                    chosen = (
                        pool
                        if quota == len(pool)
                        else rng.choice(pool, size=quota, replace=False)
                    )
                    windows, dropped_edge, dropped_base = extract_windows(
                        sequence, chosen, args.up, args.down
                    )
                    drops["edge"] += dropped_edge
                    drops["base"] += dropped_base
                    data[split]["windows"].extend(windows)
                    data[split]["labels"].extend([label] * len(windows))

    return data, totals, drops, genome_stats


# ~98% dos íntrons são canônicos (GT-AG); 0.90 dá folga para outro genoma.
MIN_DONOR_RECOVERY = 0.90


def check_strand_orientation(recovered, annotated_introns):
    """Valida a orientação de fita pela taxa de recuperação, não pela âncora GT (que é tautológica)."""
    rate = recovered / annotated_introns
    print(
        f"\nDonors recuperados sobre um GT: {recovered} de {annotated_introns} "
        f"íntrons anotados ({rate * 100:.2f}%)"
    )
    if rate < MIN_DONOR_RECOVERY:
        sys.exit(
            f"ERRO: taxa de recuperação de {rate * 100:.2f}% está abaixo do "
            f"mínimo de {MIN_DONOR_RECOVERY * 100:.0f}%.\n"
            f"       Esperado ~98% (fração de íntrons canônicos GT-AG).\n"
            f"       Uma taxa perto de zero, ou perto da metade, indica "
            f"orientação de fita ou conversão de coordenada errada."
        )


def check(data, args):
    """Falha cedo se um invariante do dataset for violado (recorte da janela, não a fita)."""
    window_length = args.up + args.down

    for split, content in data.items():
        windows = content["windows"]
        labels = content["labels"]

        if not windows:
            sys.exit(f"ERRO: split '{split}' ficou vazio.")

        bad_length = [w for w in windows if len(w) != window_length]
        if bad_length:
            sys.exit(
                f"ERRO: {len(bad_length)} janelas em '{split}' com comprimento "
                f"diferente de {window_length}."
            )

        positives = [w for w, y in zip(windows, labels) if y == 1]
        bad_gt = [w for w in positives if w[args.up : args.up + 2] != "GT"]
        if bad_gt:
            sys.exit(
                f"ERRO: {len(bad_gt)} de {len(positives)} janelas POSITIVAS em "
                f"'{split}' não têm 'GT' na posição {args.up}.\n"
                f"       Exemplo: {bad_gt[0]}\n"
                f"       Isso quase sempre significa orientação de fita ou "
                f"conversão de coordenada errada."
            )

        negatives = [w for w, y in zip(windows, labels) if y == 0]
        bad_neg = [w for w in negatives if w[args.up : args.up + 2] != "GT"]
        if bad_neg:
            sys.exit(
                f"ERRO: {len(bad_neg)} janelas NEGATIVAS em '{split}' não têm "
                f"'GT' na âncora. O pool de negativos está errado."
            )

    print("\nChecagens de sanidade: OK")
    print(f"  toda janela tem {window_length} bases")
    print(f"  toda janela (positiva e negativa) tem 'GT' na posição {args.up}")


def report_overlap(data):
    """Conta quantas janelas de teste são idênticas a alguma do treino (não é erro)."""
    train = set(data["train"]["windows"])
    test = data["test"]["windows"]
    repeated = sum(1 for window in test if window in train)
    print(
        f"\nJanelas de teste que aparecem idênticas no treino: "
        f"{repeated} de {len(test)} ({repeated / len(test) * 100:.2f}%)"
    )
    return repeated


def save(data, totals, drops, repeated, genome_stats, args):
    arrays = {}
    summary = {}
    dtype = f"S{args.up + args.down}"

    for split, content in data.items():
        windows = np.array(content["windows"], dtype=dtype)
        labels = np.array(content["labels"], dtype=np.uint8)
        order = np.random.default_rng(args.seed).permutation(len(labels))
        arrays[f"X_{split}"] = windows[order]
        arrays[f"y_{split}"] = labels[order]
        summary[split] = {
            "n": int(len(labels)),
            "n_pos": int(labels.sum()),
            "n_neg": int((labels == 0).sum()),
            "pool_pos": totals[split]["pos"],
            "pool_neg": totals[split]["neg"],
            "natural_ratio": totals[split]["neg"] / totals[split]["pos"],
        }

    meta = {
        "upstream": args.up,
        "downstream": args.down,
        "window_length": args.up + args.down,
        "donor_offset": args.up,
        "neg_pool": args.neg_pool,
        "neg_ratio": args.neg_ratio,
        "max_pos": args.max_pos,
        "eval_neg_ratio": args.eval_neg_ratio,
        "eval_max_pos": args.eval_max_pos,
        "seed": args.seed,
        "splits": SPLITS,
        "counts": summary,
        "dropped_edge": drops["edge"],
        "dropped_invalid_base": drops["base"],
        "test_windows_seen_in_train": repeated,
        "genome_stats": genome_stats,
        "source_fasta": Path(args.fasta).name,
        "source_gff3": Path(args.gff3).name,
    }
    arrays["meta"] = np.array(json.dumps(meta, ensure_ascii=False))

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)

    print("\nDataset final:")
    for split, info in summary.items():
        print(
            f"  {split:5s}  n={info['n']:7d}  "
            f"positivos={info['n_pos']:6d}  negativos={info['n_neg']:7d}  "
            f"({info['n_neg'] / max(info['n_pos'], 1):.0f}:1, "
            f"natural {info['natural_ratio']:.0f}:1)"
        )
    print(f"\nDescartadas por borda de cromossomo: {drops['edge']}")
    print(f"Descartadas por base fora de ACGT:   {drops['base']}")
    print(f"\nGravado em {output}  ({output.stat().st_size / 1e6:.1f} MB)")


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--fasta", default="data/TgrandC1074.fa.gz")
    parser.add_argument(
        "--gff3",
        default="data/TgrandC1074-Annotation-Primary_Transcripts.gff3.gz",
    )
    parser.add_argument(
        "--fasta-url", default=FASTA_URL,
        help="baixado para --fasta se o arquivo ainda não existir",
    )
    parser.add_argument(
        "--gff3-url", default=GFF3_URL,
        help="baixado para --gff3 se o arquivo ainda não existir",
    )
    parser.add_argument("--out", default="data/splice_donor.npz")
    parser.add_argument(
        "--up", type=int, default=100,
        help="bases do lado do éxon, antes do GT",
    )
    parser.add_argument(
        "--down", type=int, default=101,
        help="bases do GT até o fim da janela, dentro do íntron",
    )
    parser.add_argument(
        "--neg-pool",
        choices=["gene-body", "genome"],
        default="gene-body",
        help="de onde tirar os negativos; gene-body é o cenário de uso real",
    )
    parser.add_argument(
        "--neg-ratio", type=float, default=10.0,
        help="negativos por positivo no treino",
    )
    parser.add_argument(
        "--max-pos", type=int, default=5000,
        help="teto de positivos no treino, 0 para usar todos",
    )
    parser.add_argument(
        "--eval-neg-ratio", type=float, default=50.0,
        help="negativos por positivo em val/teste, a razão natural",
    )
    parser.add_argument(
        "--eval-max-pos", type=int, default=2000,
        help="teto de positivos em val/teste",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.up < 1 or args.down < 2:
        sys.exit("ERRO: --up deve ser >= 1 e --down >= 2 (o GT precisa caber).")
    download_if_missing(args.fasta_url, args.fasta)
    download_if_missing(args.gff3_url, args.gff3)
    data, totals, drops, genome_stats = build_dataset(args)
    check(data, args)
    repeated = report_overlap(data)
    save(data, totals, drops, repeated, genome_stats, args)


if __name__ == "__main__":
    main()
