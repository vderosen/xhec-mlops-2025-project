<div align="center">

# MLOps Project: Abalone Age Prediction

[![Python Version](https://img.shields.io/badge/python-3.10%20or%203.11-blue.svg)]()
[![Linting: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-informational?logo=pre-commit&logoColor=white)](https://github.com/artefactory/xhec-mlops-project-student/blob/main/.pre-commit-config.yaml)
</div>

## 🎯 Project Overview

Welcome to our MLOps project! In this hands-on project, we built a complete machine learning system to predict the age of abalone (a type of sea snail) using physical measurements instead of the traditional time-consuming method of counting shell rings under a microscope.

**Our Mission**: Transform a simple ML model into a production-ready system with automated training, deployment, and prediction capabilities.

## 📊 About the Dataset

Traditionally, determining an abalone's age requires:
1. Cutting the shell through the cone
2. Staining it
3. Counting rings under a microscope (very time-consuming!)

**Out Goal**: Use easier-to-obtain physical measurements (shell weight, diameter, etc.) to predict the age automatically.

📥 **Download**: Get the dataset from the [Kaggle page](https://www.kaggle.com/datasets/rodolfomendes/abalone-dataset)



## 👥 Team Information

**Team Members:**
- Soumyabrata Bairagi (roy.bensimon@polytechnique.edu)
- Samuel Rajzman (samuel.rajzman@hec.edu)
- Roy Bensimon (roy.bensimon@hec.edu)
- Vassili de Rosen (vassili.de-rosen@hec.edu)
- Adam Berdah (adam.berdah@hec.edu)

## 🛠️ Development Environment

### Prerequisites
- Python 3.10 or 3.11
- UV package manager
- Git
- GitHub account

### Complete Setup Instructions

1. **Install UV Package Manager:**
   ```bash
   # macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Windows
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. **Clone and Setup Repository:**
   ```bash
   # Clone your forked repository
   git clone https://github.com/YOUR_USERNAME/xhec-mlops-2025-project.git
   cd xhec-mlops-2025-project

   # Install all dependencies
   uv sync --extra dev

   # Install pre-commit hooks
   uv run pre-commit install
   ```

3. **Verify Installation:**
   ```bash
   # Check if everything is working
   uv run pre-commit run --all-files
   uv run ruff check .
   uv run pytest --version
   ```

### Development Workflow

**Daily Development:**
```bash
# Start your development session
uv sync --extra dev

# Run code quality checks
uv run pre-commit run --all-files

# Run tests
uv run pytest

# Run linting
uv run ruff check .
```

**Adding New Dependencies:**
```bash
# Add a new dependency
uv add <package>==<version>

# Add a dev dependency
uv add --dev <package>==<version>

# Sync environment
uv sync
```

### Code Quality Tools

**Automated Tools:**
- **Pre-commit hooks:** Run automatically on every commit
- **Ruff:** Fast Python linting and formatting
- **Black:** Code formatting (via ruff)
- **isort:** Import sorting
- **Pytest:** Testing framework

**Manual Commands:**
```bash
# Format code
uv run ruff format .

# Lint code
uv run ruff check .

# Run tests with coverage
uv run pytest --cov=.

# Check all pre-commit hooks
uv run pre-commit run --all-files
```

### CI/CD Pipeline

**Automated Checks:**
- ✅ Code quality (ruff, pre-commit)
- ✅ Testing (pytest with coverage)
- ✅ Multi-Python support (3.10, 3.11)
- ✅ Dependency management (UV)

**GitHub Actions:**
- Runs on every push and pull request
- Tests on multiple Python versions
- Generates coverage reports
- Uploads to Codecov

### Troubleshooting

**Common Issues:**

1. **UV not found:**
   ```bash
   # Add UV to your PATH
   export PATH="$HOME/.cargo/bin:$PATH"
   # Or restart your terminal
   ```

2. **Pre-commit hooks failing:**
   ```bash
   # Update pre-commit hooks
   uv run pre-commit autoupdate
   uv run pre-commit install
   ```

3. **Dependencies not installing:**
   ```bash
   # Clear UV cache and reinstall
   uv cache clean
   uv sync --extra dev
   ```

4. **Python version issues:**
   ```bash
   # Check Python version
   python --version
   # Should be 3.10 or 3.11
   ```

**Getting Help:**
- Check the [UV documentation](https://docs.astral.sh/uv/)
- Review [Pre-commit documentation](https://pre-commit.com/)
- Check GitHub Actions logs for CI issues


## 🎯 Final Deliverables Checklist

When you're done, your repository should contain:

✅ **Automated Training Pipeline**
- [ ] Prefect workflows for model training
- [ ] Separate modules for training and inference
- [ ] Reproducible model and encoder generation

✅ **Automated Deployment**
- [ ] Prefect deployment for regular retraining

✅ **Production API**
- [ ] Working REST API for predictions
- [ ] Pydantic input validation
- [ ] Docker containerization

✅ **Professional Documentation**
- [x] Updated README with team info
- [x] Clear setup and run instructions
- [x] Complete development environment setup
- [x] Troubleshooting guide
- [ ] All TODOs removed from code (in progress)

---

**Ready to start? Head to branch_0 and read PR_0.md for your first task! 🚀**
