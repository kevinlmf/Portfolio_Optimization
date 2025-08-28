# ML Alpha + Copula Beta Portfolio Optimization Framework

## Overview

This framework integrates **machine learning methods for Alpha extraction** with **Copula theory for Beta estimation** to build an advanced portfolio optimization system.

---

## 1. Alpha Extraction – Machine Learning Methods

### 1.1 Definition of Alpha

Alpha represents the excess return of an asset over the market benchmark:

$$
\alpha_i = R_i - \beta_i R_m - r_f
$$

where:
- $R_i$: return of asset $i$  
- $R_m$: market return  
- $\beta_i$: Beta coefficient of asset $i$  
- $r_f$: risk-free rate  

### 1.2 Machine Learning Alpha Prediction Model

**Feature Engineering:**  
Construct a multidimensional feature vector  
$$X_t = [X_1^t, X_2^t, ..., X_n^t]$$  
including:
- Technical indicators: RSI, MACD, Bollinger Bands, etc.  
- Momentum factors: short-, mid-, and long-term momentum  
- Volatility indicators: historical volatility, GARCH volatility  
- Fundamental indicators: valuation ratios, financial metrics  

**Prediction Model:**  

$$
\hat{\alpha}_t = \sum_{k=1}^K w_k f_k(X_t)
$$

where:
- $f_k$: the $k$-th base model (RF, XGBoost, LightGBM, LSTM)  
- $w_k$: model weight  

**Confidence Score:**  

$$
C_t = \frac{1}{K} \sum_{k=1}^K \max\Big(0, 1 - \frac{|\hat{\alpha}_k^t - \bar{\alpha}^t|}{\sigma_{\alpha}^t}\Big)
$$

---

## 2. Beta Estimation – Copula Methods

### 2.1 Limitations of Traditional Beta

Traditional CAPM Beta:

$$
\beta_i = \frac{Cov(R_i, R_m)}{Var(R_m)}
$$

**Limitations:**
- Assumes linear dependence  
- Ignores tail dependence  
- Does not account for non-normal distributions  

### 2.2 Copula-CVaR Beta Estimation

**Step 1: Marginal Distribution Modeling**

$$
F_i(r) = P(R_i \leq r)
$$

Use skewed-t distribution or empirical distribution functions.

**Step 2: Copula Dependence Structure**

$$
C(u_i, u_m) = P(F_i(R_i) \leq u_i, F_m(R_m) \leq u_m)
$$

Common Copulas:
- Gaussian Copula:  
  $$
  C^{Ga}(u,v) = \Phi_2(\Phi^{-1}(u), \Phi^{-1}(v); \rho)
  $$
- t-Copula:  
  $$
  C^t(u,v) = t_{\nu,\rho}(t_\nu^{-1}(u), t_\nu^{-1}(v))
  $$

**Step 3: CVaR-based Beta**

$$
\beta_i^{CVaR} = \frac{E[R_i \mid R_m \leq VaR_\alpha(R_m)]}{CVaR_\alpha(R_m)}
$$

where:
- $VaR_\alpha(R_m)$: Value-at-Risk of market returns  
- $CVaR_\alpha(R_m) = E[R_m \mid R_m \leq VaR_\alpha(R_m)]$  

---

## 3. Portfolio Optimization Framework

### 3.1 Objective Function

Utility combining Alpha and Beta:

$$
U(w) = w^T\hat{\alpha} - \frac{\gamma}{2}w^T\Sigma w - \lambda TC(w)
$$

where:
- $\hat{\alpha}$: ML-predicted Alpha vector  
- $\Sigma$: Copula-Beta risk covariance matrix  
- $\gamma$: risk aversion parameter  
- $TC(w)$: transaction costs  

### 3.2 Risk Model

**Copula-based covariance matrix:**

$$
\Sigma_{ij} = \sqrt{\sigma_i\sigma_j} \cdot \rho_{ij}^{Copula}
$$

where $\rho_{ij}^{Copula}$ comes from Copula dependence.

### 3.3 Constraints

$$
\begin{aligned}
\sum_i w_i &= 1 \quad &\text{(budget constraint)} \\
|w_i| &\leq w_{max} \quad &\text{(position limit)} \\
\sum_i |w_i - w_i^{prev}| &\leq \tau_{max} \quad &\text{(turnover limit)}
\end{aligned}
$$

---

## 4. Adaptive Optimization Mechanism

### 4.1 Dynamic Parameter Adjustment

**Adaptive Alpha weight:**

$$
w_\alpha^t = w_\alpha^{base} \cdot (1 + \eta \cdot sharpe_\alpha^{t-1})
$$

**Adaptive Beta weight:**

$$
w_\beta^t = w_\beta^{base} \cdot (1 + \eta \cdot accuracy_\beta^{t-1})
$$

### 4.2 Market Regime Detection

Use Hidden Markov Models (HMM) for regime switching:
- Bull market: higher Alpha weight  
- Bear market: higher Beta weight  
- Sideways market: balanced weights  

---

## 5. Performance Evaluation

### 5.1 Risk-Adjusted Performance Metrics

- **Sharpe Ratio:**  
  $$
  SR = \frac{E[R_p] - r_f}{\sigma_p}
  $$

- **Sortino Ratio:**  
  $$
  Sortino = \frac{E[R_p] - r_f}{\sigma_{downside}}
  $$

- **Information Ratio:**  
  $$
  IR = \frac{E[R_p] - E[R_b]}{TE}
  $$

### 5.2 Alpha Attribution

**Total Alpha Decomposition:**

$$
\alpha_{total} = \alpha_{ML} + \alpha_{selection} + \alpha_{interaction}
$$

**Beta Effectiveness Metric:**

$$
\beta_{eff} = \frac{Cov(\hat{\beta}, \beta_{realized})}{Var(\hat{\beta})}
$$

---

## 6. Theoretical Advantages

### 6.1 Compared to Traditional Methods

1. **Nonlinear relationships** captured by ML models  
2. **Tail risk modeling** through Copula dependence  
3. **Dynamic adaptiveness** with self-adjusting weights  
4. **Feature richness** leveraging diverse information sources  

### 6.2 Innovation Points

1. **Alpha-Beta decoupling**: separate modeling of returns and risks  
2. **Multi-scale modeling**: short-term Alpha + long-term Beta  
3. **Confidence-weighted allocation**  
4. **Cross-asset learning** using dependence structures  

---

## References

1. Fama, E. F., & French, K. R. (1996). *Multifactor explanations of asset pricing anomalies.*  
2. Joe, H. (2014). *Dependence Modeling with Copulas.*  
3. Rockafellar, R. T., & Uryasev, S. (2000). *Optimization of Conditional Value-at-Risk.*  
4. Breiman, L. (2001). *Random forests.*  
