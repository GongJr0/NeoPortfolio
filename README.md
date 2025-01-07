<a href="https://gongjr0.github.io/NeoPortfolio/">
<img src="docs/logo.svg"></img>
</a>
<h4 align="right">by Güney Kıymaç</h4>
<hr> </hr>

<div align="right">
    <table align="right" border-collapse="collapse" width="100%" border-style="hidden!important">
        <thead>
            <tr>
                <th border-collapse="collapse" border-style="hidden"><a href="https://gongjr0.github.io/NeoPortfolio/">Documentation</a></th>
                <th><a href="https://pypi.org/project/NeoPortfolio/">PyPI</a></th>
            </tr>
        </thead>
    </table>
</div>
<br></br>

## 🌟 Highlights

- Utilize MPT with added ML features
- Automatically select the best portfolio from a market
- Generate HTML reports detailing portfolios


## ℹ️ Overview
NeoPortfolio extends Modern Portfolio Theory (MPT) with NLP and ML features. The package is geared to reduce the friction 
in portfolio selection and management by maintaining simplicity in its user-facing interface. Optimize a pre-determined
portfolio or let the package automatically select the best portfolio; either way, the results are one function call
away!
### ✍️ Authors
For now, just me! Güney Kıymaç. A finance student and a self-taught data science enthusiast.  Contributions and suggestions
are always welcome. 

## 🚀 Usage
As mentioned, the package is designed for simple use. Define your investment preferences on class declaration, and
make a single function call to get the results.

```python
from NeoPortfolio import nCrOptimize
opt = nCrOptimize(
  market="^GSPC", # S&P 500
  n=5, # Number of assets in the portfolio
  target_return=0.1,
  horizon=21,
  lookback=252,
  max_pool_size=100, # Maximum number of portfolios to consider
  api_key_path="path/to/your/api/key.env", # NewsAPI key (has free tier)
  api_var_name="YOU_KEY_VAR"
)
opt.optimize_space(bounds=(0.05, 0.7))
```

## ⬇️ Installation
NeoPortfolio is available on PyPI, so you can access it with `pip`. __Python 3.12+__ is required for NeoPortfolio.

```bash
python -m pip install NeoPortfolio
```
Dependencies will be installed during the pip installation process but PyTorch can cause errors depending on your system and 
environment. If you encounter any issues, please refer to the [PyTorch installation guide](https://pytorch.org/get-started/locally/).
You only the CPU compute platform and `torchvision` or `torchaudio` are not required for this package. 
(The commands copied from the guide will install all 3 packages unless you remove them.)

## 💭 Feedback and Contributing
Feel free to use the [Discussions](https://github.com/GongJr0/NeoPortfolio/discussions) and [Issues](https://github.com/GongJr0/NeoPortfolio/issues) tabs for feedback and suggestions. As NeoPortfolio is a small scale 
project, there aren't guidelines for contributing. Shoot your suggestions and we'll work on them!