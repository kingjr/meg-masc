import mne
import mne_bids
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, scale
from tqdm import trange
from wordfreq import zipf_frequency
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import matplotlib

matplotlib.use("Agg")


mne.set_log_level(False)


class PATHS:
    path_file = Path("./data_path.txt")
    if not path_file.exists():
        data = Path(input("data_path?"))
        assert data.exists()
        with open(path_file, "w") as f:
            f.write(str(data) + "\n")
    with open(path_file, "r") as f:
        data = Path(f.readlines()[0].strip("\n"))

    assert data.exists()

    bids = data / "bids_anonym"


def segment(raw):
    # preproc annotations
    meta = list()
    for annot in raw.annotations:
        d = eval(annot.pop("description"))
        for k, v in annot.items():
            assert k not in d.keys()
            d[k] = v
        meta.append(d)
    meta = pd.DataFrame(meta)
    meta["intercept"] = 1.0

    # compute voicing
    phonemes = meta.query('kind=="phoneme"')
    assert len(phonemes)
    for ph, d in phonemes.groupby("phoneme"):
        ph = ph.split("_")[0]
        match = ph_info.query("phoneme==@ph")
        assert len(match) == 1
        meta.loc[d.index, "voiced"] = match.iloc[0].phonation == "v"

    # compute word frquency and merge w/ phoneme
    meta["is_word"] = False
    words = meta.query('kind=="word"').copy()
    assert len(words) > 10
    # assert np.all(meta.loc[words.index + 1, "kind"] == "phoneme")
    meta.loc[words.index + 1, "is_word"] = True
    wfreq = lambda x: zipf_frequency(x, "en")  # noqa
    meta.loc[words.index + 1, "wordfreq"] = words.word.apply(wfreq).values

    meta = meta.query('kind=="phoneme"')
    assert len(meta.wordfreq.unique()) > 2

    # segment
    events = np.c_[
        meta.onset * raw.info["sfreq"], np.ones((len(meta), 2))
    ].astype(int)

    epochs = mne.Epochs(
        raw,
        events,
        tmin=-0.200,
        tmax=0.6,
        decim=10,
        baseline=(-0.2, 0.0),
        metadata=meta,
        preload=True,
        event_repeated="drop",
    )

    # threshold
    th = np.percentile(np.abs(epochs._data), 95)
    epochs._data[:] = np.clip(epochs._data, -th, th)
    epochs.apply_baseline()
    th = np.percentile(np.abs(epochs._data), 95)
    epochs._data[:] = np.clip(epochs._data, -th, th)
    epochs.apply_baseline()
    return epochs


def decod(X, y, meta, times):
    assert len(X) == len(y) == len(meta)
    meta = meta.reset_index()

    y = scale(y[:, None])[:, 0]
    if len(set(y[:1000])) > 2:
        y = y > np.nanmedian(y)

    # define data
    model = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
    cv = KFold(5, shuffle=True, random_state=0)

    # fit predict
    n, nchans, ntimes = X.shape
    preds = np.zeros((n, ntimes))
    for t in trange(ntimes):
        preds[:, t] = cross_val_predict(
            model, X[:, :, t], y, cv=cv, method="predict_proba"
        )[:, 1]

    # score
    out = list()
    for label, m in meta.groupby("label"):
        Rs = correlate(y[m.index, None], preds[m.index])
        for t, r in zip(times, Rs):
            out.append(dict(score=r, time=t, label=label, n=len(m.index)))
    return pd.DataFrame(out)


def correlate(X, Y):
    if X.ndim == 1:
        X = X[:, None]
    if Y.ndim == 1:
        Y = Y[:, None]
    X = X - X.mean(0)
    Y = Y - Y.mean(0)

    SX2 = (X**2).sum(0) ** 0.5
    SY2 = (Y**2).sum(0) ** 0.5
    SXY = (X * Y).sum(0)
    return SXY / (SX2 * SY2)


def plot(result):
    fig, ax = plt.subplots(1, figsize=[6, 6])
    sns.lineplot(x="time", y="score", data=result, hue="label", ax=ax)
    ax.axhline(0, color="k")
    return fig


ph_info = pd.read_csv("phoneme_info.csv")
subjects = pd.read_csv(PATHS.bids / "participants.tsv", sep="\t")
subjects = subjects.participant_id.apply(lambda x: x.split("-")[1]).values


def _get_epochs(subject):
    all_epochs = list()
    for session in range(2):
        for task in range(4):
            print(".", end="")
            bids_path = mne_bids.BIDSPath(
                subject=subject,
                session=str(session),
                task=str(task),
                datatype="meg",
                root=PATHS.bids,
            )
            try:
                raw = mne_bids.read_raw_bids(bids_path)
            except FileNotFoundError:
                print("missing", subject, session, task)
                continue
            raw = raw.pick_types(
                meg=True, misc=False, eeg=False, eog=False, ecg=False
            )

            raw.load_data().filter(0.5, 30.0, n_jobs=1)
            epochs = segment(raw)
            epochs.metadata["half"] = np.round(
                np.linspace(0, 1.0, len(epochs))
            ).astype(int)
            epochs.metadata["task"] = task
            epochs.metadata["session"] = session

            all_epochs.append(epochs)
    if not len(all_epochs):
        return
    epochs = mne.concatenate_epochs(all_epochs)
    m = epochs.metadata
    label = (
        "t"
        + m.task.astype(str)
        + "_s"
        + m.session.astype(str)
        + "_h"
        + m.half.astype(str)
    )
    epochs.metadata["label"] = label
    return epochs


def _decod_one_subject(subject):
    epochs = _get_epochs(subject)
    if epochs is None:
        return
    # words
    words = epochs["is_word"]
    evo = words.average()
    fig_evo = evo.plot(spatial_colors=True, show=False)

    X = words.get_data() * 1e13
    y = words.metadata["wordfreq"].values

    results = decod(X, y, words.metadata, words.times)
    results["subject"] = subject
    results["contrast"] = "wordfreq"

    fig_decod = plot(results)

    # Phonemes
    phonemes = epochs["not is_word"]
    evo = phonemes.average()
    fig_evo_ph = evo.plot(spatial_colors=True, show=False)

    X = phonemes.get_data() * 1e13
    y = phonemes.metadata["voiced"].values

    results_ph = decod(X, y, phonemes.metadata, phonemes.times)
    results_ph["subject"] = subject
    results_ph["contrast"] = "voiced"
    fig_decod_ph = plot(results_ph)

    return fig_evo, fig_decod, results, fig_evo_ph, fig_decod_ph, results_ph


if __name__ == "__main__":
    report = mne.Report()

    # decoding
    all_results = list()
    results = list()
    for subject in subjects:
        print(subject)

        out = _decod_one_subject(subject)
        if out is None:
            continue

        (
            fig_evo,
            fig_decod,
            results,
            fig_evo_ph,
            fig_decod_ph,
            results_ph,
        ) = out

        report.add_figure(fig_evo, subject, tags="evo_word")
        report.add_figure(fig_decod, subject, tags="word")
        report.add_figure(fig_evo_ph, subject, tags="evo_phoneme")
        report.add_figure(fig_decod_ph, subject, tags="phoneme")

        report.save("decoding.html", open_browser=False, overwrite=True)

        all_results.append(results)
        all_results.append(results_ph)
        print("done")

    pd.concat(all_results, ignore_index=True).to_csv("decoding_results.csv")
