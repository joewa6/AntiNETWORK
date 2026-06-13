"""Compute DeepSP features for antibody sequences. Run under the deepsp conda env."""
import os, sys, tempfile, argparse

# Ensure ANARCI (installed in this env) is on PATH for os.system calls
_env_bin = os.path.dirname(sys.executable)
os.environ["PATH"] = _env_bin + ":" + os.environ.get("PATH", "")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import pandas as pd
from pathlib import Path
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from keras.models import model_from_json

WEIGHTS = Path('/tmp/abdev-benchmark/models/deepsp_ridge/model_weights')

H_INCL = [str(i) for i in range(1, 111)] + \
         ['111','111A','111B','111C','111D','111E','111F','111G','111H'] + \
         ['112I','112H','112G','112F','112E','112D','112C','112B','112A','112'] + \
         [str(i) for i in range(113, 129)]
L_INCL = [str(i) for i in range(1, 128)]
H_DICT = {p: i for i, p in enumerate(H_INCL)}
L_DICT = {p: i for i, p in enumerate(L_INCL)}
AA_D = {'A':0,'C':1,'D':2,'E':3,'F':4,'G':5,'H':6,'I':7,'K':8,'L':9,'M':10,
        'N':11,'P':12,'Q':13,'R':14,'S':15,'T':16,'V':17,'W':18,'Y':19,'-':20}

FEATURE_NAMES = [
    'SAP_pos_CDRH1','SAP_pos_CDRH2','SAP_pos_CDRH3','SAP_pos_CDRL1','SAP_pos_CDRL2',
    'SAP_pos_CDRL3','SAP_pos_CDR','SAP_pos_Hv','SAP_pos_Lv','SAP_pos_Fv',
    'SCM_neg_CDRH1','SCM_neg_CDRH2','SCM_neg_CDRH3','SCM_neg_CDRL1','SCM_neg_CDRL2',
    'SCM_neg_CDRL3','SCM_neg_CDR','SCM_neg_Hv','SCM_neg_Lv','SCM_neg_Fv',
    'SCM_pos_CDRH1','SCM_pos_CDRH2','SCM_pos_CDRH3','SCM_pos_CDRL1','SCM_pos_CDRL2',
    'SCM_pos_CDRL3','SCM_pos_CDR','SCM_pos_Hv','SCM_pos_Lv','SCM_pos_Fv',
]


def load_models():
    out = {}
    for name in ['SAPpos', 'SCMneg', 'SCMpos']:
        with open(WEIGHTS / f'Conv1D_regression{name}.json') as f:
            m = model_from_json(f.read())
        m.load_weights(str(WEIGHTS / f'Conv1D_regression_{name}.h5'))
        out[name] = m
    return out


def align_batch(names, vh_seqs, vl_seqs, tmpdir):
    tmpdir = Path(tmpdir)
    with open(tmpdir / 'seq_H.fasta', 'w') as f:
        SeqIO.write([SeqRecord(Seq(s), id=n, name='', description='')
                     for n, s in zip(names, vh_seqs)], f, 'fasta')
    with open(tmpdir / 'seq_L.fasta', 'w') as f:
        SeqIO.write([SeqRecord(Seq(s), id=n, name='', description='')
                     for n, s in zip(names, vl_seqs)], f, 'fasta')
    os.system(f'ANARCI -i {tmpdir}/seq_H.fasta -o {tmpdir}/seq_H -s imgt -r heavy --csv 2>/dev/null')
    os.system(f'ANARCI -i {tmpdir}/seq_L.fasta -o {tmpdir}/seq_L -s imgt -r light --csv 2>/dev/null')

    h_df = pd.read_csv(tmpdir / 'seq_H_H.csv')
    l_csv = next((tmpdir / c for c in ['seq_L_KL.csv','seq_L_K.csv','seq_L_L.csv']
                  if (tmpdir / c).exists()), None)
    if l_csv is None:
        raise FileNotFoundError(f"No light chain CSV in {tmpdir}")
    l_df = pd.read_csv(l_csv)

    aligned = []
    for i in range(len(names)):
        h_tmp = ['-'] * 145
        l_tmp = ['-'] * 127
        for col in h_df.columns:
            if col in H_DICT:
                v = h_df.iloc[i][col]
                h_tmp[H_DICT[col]] = v if isinstance(v, str) and v not in ('nan','') else '-'
        for col in l_df.columns:
            if col in L_DICT:
                v = l_df.iloc[i][col]
                l_tmp[L_DICT[col]] = v if isinstance(v, str) and v not in ('nan','') else '-'
        aligned.append(''.join(h_tmp + l_tmp))
    return aligned


def encode_batch(aligned):
    X = np.zeros((len(aligned), 272, 21), dtype='float32')
    for i, seq in enumerate(aligned):
        for j, c in enumerate(seq):
            X[i, j, AA_D.get(c, 20)] = 1
    return X


def compute_batch(models, names, vh_seqs, vl_seqs):
    with tempfile.TemporaryDirectory() as tmp:
        aligned = align_batch(names, vh_seqs, vl_seqs, tmp)
    X = encode_batch(aligned)
    sap = models['SAPpos'].predict(X, verbose=0)
    neg = models['SCMneg'].predict(X, verbose=0)
    pos = models['SCMpos'].predict(X, verbose=0)
    feats = np.concatenate([sap, neg, pos], axis=1)
    return pd.DataFrame(feats, columns=FEATURE_NAMES).assign(antibody_name=names)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seqs',   required=True, help='CSV with antibody_name, vh_protein_sequence, vl_protein_sequence')
    parser.add_argument('--out',    required=True, help='Output CSV path')
    parser.add_argument('--batch',  type=int, default=50)
    args = parser.parse_args()

    seqs = pd.read_csv(args.seqs)[['antibody_name','vh_protein_sequence','vl_protein_sequence']]
    out_path = Path(args.out)

    if out_path.exists():
        existing = pd.read_csv(out_path)
        # Only treat rows with actual computed features as done (skip NaN failures)
        valid = existing[existing[FEATURE_NAMES[0]].notna()]
        done = set(valid['antibody_name'])
        seqs = seqs[~seqs['antibody_name'].isin(done)]
        print(f"Resuming: {len(done)} valid, {len(seqs)} remaining", flush=True)
        results = [valid] if not valid.empty else []
    else:
        results = []

    print(f"Loading DeepSP models...", flush=True)
    models = load_models()
    print("Models loaded.", flush=True)
    rows = list(seqs.itertuples(index=False))
    batches = [rows[i:i+args.batch] for i in range(0, len(rows), args.batch)]

    for k, batch in enumerate(batches):
        names   = [r.antibody_name for r in batch]
        vh_seqs = [r.vh_protein_sequence for r in batch]
        vl_seqs = [r.vl_protein_sequence for r in batch]
        try:
            df = compute_batch(models, names, vh_seqs, vl_seqs)
            results.append(df)
            print(f"Batch {k+1}/{len(batches)} done ({names[0]})", flush=True)
        except Exception as e:
            print(f"Batch {k+1} FAILED: {e}", flush=True)
            results.append(pd.DataFrame({'antibody_name': names,
                                         **{c: float('nan') for c in FEATURE_NAMES}}))

    out = pd.concat(results, ignore_index=True).drop_duplicates('antibody_name')
    out.to_csv(out_path, index=False)
    ok = out[FEATURE_NAMES[0]].notna().sum()
    print(f"Done. {ok}/{len(out)} complete → {out_path}", flush=True)


if __name__ == '__main__':
    main()
