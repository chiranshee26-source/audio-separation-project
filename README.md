<div align="center">

# Unmix the Signal

**Recovering clean sounds from tangled recordings with Independent Component Analysis, written from scratch in NumPy.**

[![tests](https://github.com/chiranshee26-source/audio-separation-project/actions/workflows/test.yml/badge.svg)](https://github.com/chiranshee26-source/audio-separation-project/actions/workflows/test.yml)
![python](https://img.shields.io/badge/python-3.11-3776ab)
![no sklearn ICA](https://img.shields.io/badge/ICA-from%20scratch-5fd6b4)

### [Try the live demo](https://chiranshee26-source.github.io/audio-separation-project/)

</div>

---

## The idea

Put three microphones in a room where a person is talking, music is playing and rain is falling. Each microphone records a *different blend* of the same three sounds. Can you get the three original sounds back, knowing nothing about how they were blended?

Yes, if the sounds are statistically independent and non-Gaussian. This project does it with no ICA library: centering, PCA whitening and FastICA are all implemented by hand.

```
  sources S  ──(unknown mixing matrix A)──▶  recordings X = A·S
                                                   │
                         center ─▶ whiten (PCA) ─▶ FastICA
                                                   │
                                  recovered sources Ŝ  ≈  S
```

## Results

Three real clips (speech, music, rain and wind), 16 kHz, 10 s each, mixed into three recordings with a random matrix (condition number 12.4).

| Source  | Best raw microphone (SI-SDR) | After ICA (SI-SDR) | Correlation with original |
|---------|-----------------------------:|-------------------:|--------------------------:|
| Speech  |                      2.40 dB |       **58.37 dB** |                    1.0000 |
| Music   |                      1.88 dB |       **41.10 dB** |                    1.0000 |
| Ambient |                     −5.60 dB |       **38.36 dB** |                    0.9999 |

SI-SDR is scale-invariant signal-to-distortion ratio. Higher is better, and the "best raw microphone" column is the baseline you get by doing nothing.

Listen to the separated audio in [`outputs/`](outputs/) and compare it with the mixtures in `data/mixed/`.

## Live demo

The demo in [`docs/index.html`](docs/index.html) is a single static page with no backend and no database. It runs the same pipeline in JavaScript, in your browser:

- Pick a scene, press play on the mixed recordings, then watch the dots untangle.
- Edit the mixing matrix yourself, swap a source for Gaussian static, and drag the noise slider to see where separation breaks.
- Upload your own three clips (or three mixed recordings) of up to 10 seconds each.

The demo scenes are synthetic sounds generated in the page, so it works offline. The Python code is the reference implementation, and the numbers in the table above come from Python on real audio.

## Run it yourself

```powershell
git clone https://github.com/chiranshee26-source/audio-separation-project.git
cd audio-separation-project
python -m venv .venv
.venv\Scripts\Activate.ps1          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python src/make_mixtures.py         # mix the sources with a random matrix
python src/step3_center.py          # center
python src/step4_whiten.py          # PCA whitening
python src/step5_ica.py             # FastICA
python src/step6_export.py          # write separated .wav files
python src/step7_evaluate.py        # SI-SDR + correlation -> outputs/results.csv
pytest                              # unit tests
```

Figures and the full walkthrough are in [`notebooks/results.ipynb`](notebooks/results.ipynb).

## How it works

| Step | File | What it does |
|------|------|--------------|
| 1. Mix | `src/make_mixtures.py` | Builds `X = A·S` with a random positive `A`, redrawn until its condition number is below 20 so the problem is well posed. |
| 2. Center | `src/preprocessing.py` | Subtracts each recording's mean. |
| 3. Whiten | `src/preprocessing.py` | Eigendecomposition of the covariance, `V = D^(−1/2)·Eᵀ`, so the data has identity covariance. |
| 4. ICA | `src/fastica.py` | FastICA fixed-point iteration with a `tanh` contrast function and deflation (Gram–Schmidt). |
| 5. Evaluate | `src/metrics.py` | Matches estimates to originals with the Hungarian algorithm, then scores SI-SDR and correlation. |

## Why this works

**Why ICA and not PCA?** PCA finds uncorrelated directions, which only uses second-order statistics (covariance). Once data is whitened it is uncorrelated with unit variance, and it stays that way under *any* rotation. So second-order statistics cannot tell you which rotation is the right one. ICA uses higher-order statistics to pick it.

**What does whitening do?** It removes everything covariance can explain. After whitening, the mixing matrix is just a rotation, so ICA only has to find `n(n−1)/2` angles instead of `n²` free numbers. It also makes the search well conditioned.

**Why non-Gaussianity?** By the central limit theorem, a sum of independent signals is more Gaussian than the signals themselves. So we look for the projection that is *least* Gaussian (measured with kurtosis or negentropy), and that projection is a single source. If the sources were Gaussian, every rotation would look equally Gaussian and nothing could be recovered. ICA tolerates at most one Gaussian source.

**Why these metrics?** ICA cannot recover the order, sign or loudness of sources. So estimates are first matched to the originals by absolute correlation, and SI-SDR is used because it ignores scale. The baseline is the best raw microphone, so the numbers show actual improvement.

## When it does not work

ICA is not magic. It needs:

- at least as many simultaneous recordings as sources,
- sources that are independent and not Gaussian,
- instantaneous mixing. Echo and reverb (convolutive mixing) break the model,
- low recording noise. Separation quality falls steadily as noise rises, and the demo shows this with an SNR sweep.

## Audio sources

All clips were trimmed to 10 s and resampled to 16 kHz mono.

| Role | Title | Author | Source | License |
|------|-------|--------|--------|---------|
| Speech  | `orange.wav` ("Bedtime Story" pack) | PacificSea | <!-- TODO: paste Freesound URL --> | <!-- TODO: license --> |
| Music   | How Dare You (Instrumental) | Casonika | <!-- TODO: paste URL --> | CC BY-NC |
| Ambient | Rain and Wind (medium strength) | upkey | <!-- TODO: paste Freesound URL --> | <!-- TODO: license --> |

The music is licensed non-commercial, so this repository is for education and portfolio use only.

## Project layout

```
├── src/                 core code (preprocessing, FastICA, metrics, step scripts)
├── data/
│   ├── sources/         the three clean clips
│   └── mixed/           generated mixtures and the true A and S
├── outputs/             separated audio, figures, results.csv
├── notebooks/           results walkthrough with plots
├── tests/               pytest checks for centering, whitening and ICA
├── docs/                the live demo (GitHub Pages)
└── .github/workflows/   CI on Ubuntu, Windows and macOS
```

## Tests

`tests/test_ica.py` checks that centering gives zero mean, that whitening gives identity covariance, and that FastICA recovers synthetic sources (|correlation| > 0.99) across many random seeds. CI runs them on three operating systems.

## Roadmap

- PCA-versus-ICA comparison on the same mixtures, with plots
- Separation quality across SNR levels in Python (already in the web demo)
- Real recordings from multiple devices in the same room

## Tech

Python 3.11, NumPy, SciPy, Matplotlib, soundfile, Jupyter. The demo is plain HTML, Canvas and the Web Audio API.

## License

Code: MIT. Audio clips keep their original licenses (see above).
