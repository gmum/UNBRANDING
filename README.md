# From Unlearning to UNBRANDING: A Benchmark for Trademark‑Safe Text‑to‑Image Generation


[![arXiv](https://img.shields.io/badge/arXiv-2512.13953-b31b1b.svg)](https://arxiv.org/abs/2512.13953)
[![Website](https://img.shields.io/badge/Website-Project%20Page-22c55e.svg)](https://gmum.github.io/UNBRANDING/)
[![Stars gmum/VeGaS](https://img.shields.io/github/stars/gmum/UNBRANDING?style=social)](https://gmum.github.io/UNBRANDING)

![Teaser](images/teaser_ub_v4.png)

> Benchmark and task for trademark‑safe text‑to‑image generation. We introduce unbranding — fine‑grained removal of both explicit logos and implicit trade‑dress while preserving object and scene semantics — together with a dataset and a VLM‑QA evaluation protocol.


## Authors
Dawid Malarz, Artur Kasymov, Filip Manjak, Maciej Zięba, Przemysław Spurek

## Abstract
Modern text‑to‑image diffusion models can faithfully reproduce trademarks. Prior unlearning works target general concepts (e.g., styles, celebrities) and miss brand‑specific identifiers. Brand recognition is multi‑dimensional, spanning explicit logos and distinctive structural features (e.g., a car’s front grille). We define unbranding as the fine‑grained removal of both trademarks and subtle trade‑dress while preserving semantic coherence. We introduce a benchmark dataset and a VLM‑based QA metric that probes for both explicit and implicit brand signals, going beyond classic logo detectors. Our analysis shows newer models tend to synthesize brands more readily, underscoring the urgency of unbranding. Results validated by our metric indicate that unbranding is a distinct, practically relevant problem requiring specialized techniques.

## Highlights
- Task: fine‑grained removal of brand identifiers while preserving object fidelity.
- Benchmark: dataset spanning explicit logos and trade‑dress cues.
- Metric: VLM‑based QA probing both explicit and implicit brand signals.
- Motivation: newer models (e.g., SDXL, FLUX) reproduce brands more readily than older ones.

## Results Snapshot
Trade‑off between fidelity and removal. Baseline preserves structure but often fails to remove brands; ESD removes brands but alters semantics. Effective unbranding must achieve both.

![Trade‑off](images/fig4.png)

## Reproducibility Checklist
- Environment: pin model versions; fix seeds for sampling; document prompts.
- Evaluation: run VLM‑QA scripts with provided templates; report controls.
- Artifacts: configs and sample prompts for quick verification.

## Citation
Please cite our work if you find it useful:

```
@misc{malarz2025unlearningunbrandingbenchmarktrademarksafe,
  title={From Unlearning to UNBRANDING: A Benchmark for Trademark-Safe Text-to-Image Generation},
  author={Dawid Malarz and Artur Kasymov and Filip Manjak and Maciej Zięba and Przemysław Spurek},
  year={2025},
  eprint={2512.13953},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2512.13953},
}
```

## Acknowledgments
The project page is adapted from the Academic Project Page Template (inspired by Nerfies). Images in this repository are for research/illustration.
