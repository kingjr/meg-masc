# MEG-MASC: a high-quality magneto-encephalography dataset for evaluating natural speech processing.

Laura Gwilliams, Graham Flick, Alec Marantz, Liina Pylkkänen, David Poeppel, Jean-Rémi King

- [Paper](https://arxiv.org)
- [Data](https://osf.io/ag3kj/)
- [Code](https://github.com/kingjr/meg-masc)

## Abstract
The "MEG-MASC" dataset provides a curated set of raw magnetoencephalography (MEG) recordings of 27 English speakers who listened to two hours of naturalistic stories. Each participant performed two identical sessions, involving listening to four fictional stories from the Manually Annotated Sub-Corpus (MASC) intermixed with random word lists and comprehension questions. We time-stamp the onset and offset of each word and phoneme in the metadata of the recording, and organize the dataset according to the 'Brain Imaging Data Structure' (BIDS). This data collection provides a suitable benchmark to large-scale encoding and decoding analyses of temporally-resolved brain responses to speech. We provide the Python code to replicate several validations analyses of the MEG evoked related fields such as the temporal decoding of phonetic features and word frequency. All code and MEG, audio and text data are publicly available to keep with best practices in transparent and reproducible research.

## Steps
- [1] Download the code of this repository.
- [2] Download the data from the OSF website. Both part 1 (MEG-MASC) and part 2 (MEG-MASC 2).
- [3] Merge them into one folder called "bids_anonym"
- [4] Place that folder into your copy of this repository.

## Please cite
@article{gwilliams2020neural,
  title={Neural dynamics of phoneme sequencing in real speech jointly encode order and invariant content},
  author={Gwilliams, Laura and King, Jean-Remi and Marantz, Alec and Poeppel, David},
  journal={BioRxiv},
  year={2020},
  publisher={Cold Spring Harbor Laboratory}
}
