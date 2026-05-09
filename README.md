# Physics-Informed-GPR-Biogas-Yield-Prediction-From-Organic-Waste
A machine learning framework that predicts biogas yield from organic waste by integrating biochemical and thermodynamic principles — Monod kinetics, Arrhenius temperature dependence, and pH-activity relations — into a Gaussian Process Regression model, achieving R² of 0.968 with only 120 samples.

---

## 📄 Paper

**"A Hybrid Predictive Framework using Machine Learning Models for Optimizing Bio-Energy Generation from Organic Waste"**  
Aditya Roy, Aparijit Chakraborty, Tathagata Bhattacharjee, Hariharan R, Rajakani M  
*SRM Institute of Science and Technology, Kattankulathur, Chennai, India*  
*Published / Under Review — Springer*

---

## 📌 Overview

With the increasing trend in organic waste generation and the growing demand for sustainable energy, improving biogas production efficiency has become critical. Traditional anaerobic digestion systems rely on fixed feedstock ratios and operating conditions, limiting performance.

This project proposes a **Physics-Informed Gaussian Process Regression (PI-GPR)** framework that integrates biochemical and thermodynamic principles directly into the kernel function of a Gaussian Process model — enabling accurate, uncertainty-aware predictions of biogas yield from physicochemical parameters.

---

## 🧠 Key Contributions

- **PI-GPR Framework** — A novel kernel that integrates Monod kinetics, Arrhenius temperature dependence, pH-activity coupling, and microbial lifecycle dynamics into the covariance function of a Gaussian Process.
- **Physics-Constrained Feature Engineering** — Four physics-derived features computed from raw inputs to encode domain knowledge:
  - `monod(VS)` — Substrate utilization rate
  - `arrhenius(TS)` — Temperature-dependent reaction rate (via TS proxy)
  - `ph_activity(pH)` — Enzyme activity window
  - `lifecycle(Day)` — Gompertz-based microbial production phase
- **Feedstock Optimization** — Genetic Algorithm (PyGAD) optimization over the PI-GPR landscape, identifying the optimal feedstock ratio.
- **Robust Evaluation** — 5-fold cross-validation on a 120-sample dataset to prevent overfitting.

---

## 📊 Results

| Metric | PI-GPR | Standard RBF GP | Matérn GP |
|--------|--------|-----------------|-----------|
| R² (Test) | **0.968** | 0.941 | 0.953 |
| RMSE (cm³/day) | **4.21** | 6.87 | 5.63 |
| Cross-Val R² (avg) | **0.968** | — | — |

**Optimal Feedstock Mix:** 49% Food Waste + 51% Weed Biomass → **164.57 cm³/day**

---

## 🗂️ Repository Structure

```
├── Watts.ipynb                  # Main notebook: model training, evaluation, optimization
├── biogas_ml_dataset.csv        # Dataset (120 samples, 7 features + target)
├── physics_functions.png        # Visualization of physics-informed functions
├── requirements.txt             # Python dependencies
└── README.md
```

---

## ⚙️ Features Used

| Feature | Description |
|---------|-------------|
| `TS` | Total Solids (%) — proxy for operating temperature |
| `VS` | Volatile Solids (%) — substrate concentration |
| `pH` | pH of the digestion mixture |
| `C_N` | Carbon-to-Nitrogen ratio |
| `Food_pct` | Percentage of food waste in feedstock |
| `Weed_pct` | Percentage of weed biomass in feedstock |
| `Day` | Digestion day |
| `Biogass_Yield` | **Target** — Biogas yield (cm³/day) |

---

## 🔬 Physics-Informed Kernel

The custom kernel combines:

```
K(x, x') = σ²_data · K_RBF(x, x')        ← data-driven component (60%)
          + σ²_physics · K_physics(x, x')  ← physics component (40%)
          + σ²_noise · δ(x, x')            ← noise term
```

Where `K_physics` encodes similarity based on physics-derived features:

```
φ(x) = [monod(VS), arrhenius(TS), ph_activity(pH), lifecycle(Day)]
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Requirements

```
numpy
pandas
matplotlib
seaborn
scipy
scikit-learn
pygad
```

### Running the Notebook

1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/physics-informed-gpr-biogas-yield-prediction-from-organic-waste.git
   cd physics-informed-gpr-biogas-yield-prediction-from-organic-waste
   ```

2. Place `biogas_ml_dataset.csv` in the project root (or update the path in the notebook).

3. Open and run the notebook:
   ```bash
   jupyter notebook Watts.ipynb
   ```

> **Note:** The notebook was originally developed on Google Colab. If running locally, remove or replace the `drive.mount` cell and update the CSV path accordingly.

---

## 📐 Model Architecture

```
Input Features (7)
       │
       ▼
StandardScaler
       │
       ├──────────────────────────────┐
       ▼                              ▼
Physics Features φ(x)          Scaled Features x
[Monod, Arrhenius,             (data-driven component)
 pH-activity, Lifecycle]
       │                              │
       └──────────┬───────────────────┘
                  ▼
     PhysicsInformedBioEnergyKernel
                  │
                  ▼
    GaussianProcessRegressor
                  │
          ┌───────┴────────┐
          ▼                ▼
     Predicted          Uncertainty
      Yield              Estimate
    (cm³/day)           (σ bounds)
```

---

## 🧬 Physics Equations

**Monod Kinetics:**
$$\mu(S) = \mu_{max} \cdot \frac{S}{K_s + S}$$

**Arrhenius Temperature Dependence:**
$$k(T) = \exp\left(-\frac{E_a}{R}\left(\frac{1}{T} - \frac{1}{T_{ref}}\right)\right)$$

**pH-Activity Coupling:**
$$f(pH) = \exp\left(-0.5 \cdot \left(\frac{pH - pH_{opt}}{\sigma_{pH}}\right)^2\right)$$

**Microbial Lifecycle (Modified Gompertz):**
$$f(t) = \left(1 - e^{-\alpha(t - t_{lag})}\right) \cdot e^{-\beta \cdot \max(0,\ t - t_{decline})}$$

---

## 📚 References

1. Raissi, M., Perdikaris, P., & Karniadakis, G.E. (2019). Physics-informed neural networks. *Journal of Computational Physics*.
2. Wang, Xu, & Zhang — Physics-informed Gaussian process regression for anaerobic digestion modeling.
3. Rasmussen, C.E. & Williams, C.K.I. (2006). *Gaussian Processes for Machine Learning*. MIT Press.
4. Karniadakis, G.E. et al. (2021). Physics-informed machine learning. *Nature Reviews Physics*.
5. Batstone, D.J. et al. (2002). IWA Anaerobic Digestion Model No.1 (ADM1). *IWA Publishing*.

---

## 📝 Citation

If you use this code or framework in your research, please cite:

```bibtex
@article{roy2024pigpr_biogas,
  title   = {A Hybrid Predictive Framework using Machine Learning Models for 
             Optimizing Bio-Energy Generation from Organic Waste},
  author  = {Roy, Aditya and Chakraborty, Aparijit and Bhattacharjee, Tathagata 
             and Hariharan, R and Rajakani, M},
  journal = {Springer},
  year    = {2026},
  institution = {SRM Institute of Science and Technology}
}
```

---

## 📃 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🤝 Authors

| Name | Department | Institution |
|------|-----------|-------------|
| Aditya Roy | Computing Technologies | SRMIST, Chennai |
| Aparijit Chakraborty | Computing Technologies | SRMIST, Chennai |
| Tathagata Bhattacharjee | Data Science & Business Systems | SRMIST, Chennai |
| Hariharan R | Computing Technologies | SRMIST, Chennai |
| Rajakani M | Data Science & Business Systems | SRMIST, Chennai |
