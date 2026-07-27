# Table 2 — Comparison with prior fundus-to-VF studies

| Study | Data (source) | Prediction target | Pooled MAE (dB) | Mild / Mod / Severe (dB) | External validation |
|---|---|---|---|---|---|
| **This study** | GRAPE, open-access (631 records) | 24-2 raw sensitivity | **4.10** (composition-adjusted 3.27–3.65) | 2.90 / 4.06 / 8.43 | **PAPILA, r 0.75** |
| Park et al. 2026 (10) | Private, 2 tertiary hospitals (~37.9k photos) | 24-2 total deviation | 3.91 | 3.09 / 5.66 / 9.15 | None reported |
| Huang et al. 2025 (11) | Private (1,129 eyes) | Octopus 59-point sensitivity† | 3.10–4.13 | — | None reported |
| Kang et al. 2024 (9) | Private | VF image (cGAN)‡ | — | — | None reported |

**Legend.**
† Different device/point layout/dynamic range — not directly comparable to 24-2 dB.
‡ Image-translation output judged by PSNR 30.61 dB / SSIM 0.48, not clinical sensitivity error.
Mild = each location's true sensitivity ≥22 dB; moderate 15–22; severe <15. Our mild figure is
statistically equivalent to Park et al.'s (within measurement error).
